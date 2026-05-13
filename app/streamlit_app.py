import streamlit as st
import torch
import torch.nn as nn
import re
from pathlib import Path
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Works whether you run Streamlit from the project root or from app/
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_PATH = PROJECT_DIR / "dataset_analysis"/"models" / "best_seq2seq_attention_2layer.pt"
RESULTS_DIR = PROJECT_DIR / "results"
ROUGE_PATH = RESULTS_DIR / "rouge_scores_arabic_recall_focused.csv"

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


# -------------------------
# Text preprocessing
# -------------------------
arabic_diacritics = re.compile("""
                             ّ    |
                             َ    |
                             ً    |
                             ُ    |
                             ٌ    |
                             ِ    |
                             ٍ    |
                             ْ    |
                             ـ
                         """, re.VERBOSE)


def remove_diacritics(text):
    return re.sub(arabic_diacritics, "", str(text))


def normalize_arabic(text):
    text = str(text)
    text = re.sub(r"[إأآا]", "ا", text)
    # text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"گ", "ك", text)
    return text


def clean_arabic_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------------
# Model helper functions
# -------------------------
def create_encoder(input_dim, emb_dim, hidden_dim, num_layers=1, dropout=0.4, pad_idx=0):
    return {
        "embedding": nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx).to(DEVICE),
        "rnn": nn.GRU(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        ).to(DEVICE),
        "fc": nn.Linear(hidden_dim * 2, hidden_dim).to(DEVICE),
        "dropout": nn.Dropout(dropout),
        "num_layers": num_layers,
        "hidden_dim": hidden_dim
    }


def encoder_forward(encoder, src, dec_num_layers):
    embedded = encoder["dropout"](encoder["embedding"](src))
    outputs, hidden = encoder["rnn"](embedded)

    # Last encoder layer forward/backward states
    hidden_forward = hidden[-2]
    hidden_backward = hidden[-1]
    hidden_cat = torch.cat((hidden_forward, hidden_backward), dim=1)

    hidden = torch.tanh(encoder["fc"](hidden_cat))
    hidden = hidden.unsqueeze(0).repeat(dec_num_layers, 1, 1)

    return outputs, hidden


def create_attention(encoder_hidden_dim, decoder_hidden_dim):
    return {
        "attn": nn.Linear((encoder_hidden_dim * 2) + decoder_hidden_dim, decoder_hidden_dim).to(DEVICE),
        "v": nn.Linear(decoder_hidden_dim, 1, bias=False).to(DEVICE)
    }


def attention_forward(attention, hidden, encoder_outputs, mask):
    src_len = encoder_outputs.shape[1]
    hidden_last = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)

    energy = torch.tanh(
        attention["attn"](torch.cat((hidden_last, encoder_outputs), dim=2))
    )

    attention_scores = attention["v"](energy).squeeze(2)
    attention_scores = attention_scores.masked_fill(mask == 0, -1e10)

    return torch.softmax(attention_scores, dim=1)


def create_decoder(output_dim, emb_dim, encoder_hidden_dim, decoder_hidden_dim, attention, num_layers=1, dropout=0.4, pad_idx=0):
    return {
        "output_dim": output_dim,
        "attention": attention,
        "embedding": nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx).to(DEVICE),
        "rnn": nn.GRU(
            (encoder_hidden_dim * 2) + emb_dim,
            decoder_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        ).to(DEVICE),
        "fc_out": nn.Linear(
            (encoder_hidden_dim * 2) + decoder_hidden_dim + emb_dim,
            output_dim
        ).to(DEVICE),
        "dropout": nn.Dropout(dropout),
        "num_layers": num_layers,
        "hidden_dim": decoder_hidden_dim
    }


def decoder_forward(decoder, input_token, hidden, encoder_outputs, mask):
    input_token = input_token.unsqueeze(1)
    embedded = decoder["dropout"](decoder["embedding"](input_token))

    attention_weights = attention_forward(
        decoder["attention"],
        hidden,
        encoder_outputs,
        mask
    ).unsqueeze(1)

    weighted = torch.bmm(attention_weights, encoder_outputs)
    rnn_input = torch.cat((embedded, weighted), dim=2)

    output, hidden = decoder["rnn"](rnn_input, hidden)

    embedded = embedded.squeeze(1)
    output = output.squeeze(1)
    weighted = weighted.squeeze(1)

    prediction = decoder["fc_out"](
        torch.cat((output, weighted, embedded), dim=1)
    )

    return prediction, hidden


def create_mask(src, pad_idx):
    return src != pad_idx


def set_eval_mode(model):
    for component in [
        model["encoder"]["embedding"],
        model["encoder"]["rnn"],
        model["encoder"]["fc"],
        model["decoder"]["embedding"],
        model["decoder"]["rnn"],
        model["decoder"]["fc_out"],
        model["decoder"]["attention"]["attn"],
        model["decoder"]["attention"]["v"]
    ]:
        component.eval()


def infer_model_config_from_checkpoint(checkpoint):
    """Infer hidden sizes and number of GRU layers from saved weights.
    This avoids changing the model export/checkpoint.
    """
    enc_rnn = checkpoint["encoder"]["rnn"]
    dec_rnn = checkpoint["decoder"]["rnn"]

    enc_hidden_dim = enc_rnn["weight_ih_l0"].shape[0] // 3
    dec_hidden_dim = dec_rnn["weight_ih_l0"].shape[0] // 3

    enc_num_layers = len([
        k for k in enc_rnn.keys()
        if k.startswith("weight_ih_l") and "reverse" not in k
    ])

    dec_num_layers = len([
        k for k in dec_rnn.keys()
        if k.startswith("weight_ih_l") and "reverse" not in k
    ])

    return enc_hidden_dim, dec_hidden_dim, enc_num_layers, dec_num_layers


def get_token_id(vocab, token, fallback):
    return vocab.get(token, fallback)


# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)

    source_word2idx = checkpoint["source_word2idx"]
    target_word2idx = checkpoint["target_word2idx"]
    target_idx2word = checkpoint["target_idx2word"]
    config = checkpoint["config"]

    # Keep vocab/embedding sizes from checkpoint config.
    input_dim = config["input_dim"]
    output_dim = config["output_dim"]
    enc_emb_dim = config["enc_emb_dim"]
    dec_emb_dim = config["dec_emb_dim"]
    dropout = config.get("dropout", 0.4)

    # Critical fix: infer architecture from the saved weights, not from possibly incomplete config.
    enc_hidden_dim, dec_hidden_dim, enc_num_layers, dec_num_layers = infer_model_config_from_checkpoint(checkpoint)

    source_pad_idx = source_word2idx.get(PAD_TOKEN, 0)
    target_pad_idx = target_word2idx.get(PAD_TOKEN, 0)

    attention = create_attention(enc_hidden_dim, dec_hidden_dim)

    encoder = create_encoder(
        input_dim,
        enc_emb_dim,
        enc_hidden_dim,
        num_layers=enc_num_layers,
        dropout=dropout,
        pad_idx=source_pad_idx
    )

    decoder = create_decoder(
        output_dim,
        dec_emb_dim,
        enc_hidden_dim,
        dec_hidden_dim,
        attention,
        num_layers=dec_num_layers,
        dropout=dropout,
        pad_idx=target_pad_idx
    )

    encoder["embedding"].load_state_dict(checkpoint["encoder"]["embedding"])
    encoder["rnn"].load_state_dict(checkpoint["encoder"]["rnn"])
    encoder["fc"].load_state_dict(checkpoint["encoder"]["fc"])

    decoder["embedding"].load_state_dict(checkpoint["decoder"]["embedding"])
    decoder["rnn"].load_state_dict(checkpoint["decoder"]["rnn"])
    decoder["fc_out"].load_state_dict(checkpoint["decoder"]["fc_out"])
    decoder["attention"]["attn"].load_state_dict(checkpoint["decoder"]["attention_attn"])
    decoder["attention"]["v"].load_state_dict(checkpoint["decoder"]["attention_v"])

    model = {
        "encoder": encoder,
        "decoder": decoder,
        "source_pad_idx": source_pad_idx,
        "target_pad_idx": target_pad_idx,
        "enc_hidden_dim": enc_hidden_dim,
        "dec_hidden_dim": dec_hidden_dim,
        "enc_num_layers": enc_num_layers,
        "dec_num_layers": dec_num_layers,
    }

    set_eval_mode(model)

    return model, source_word2idx, target_word2idx, target_idx2word, config


# -------------------------
# Summary generation
# -------------------------
def summarise_text(
    text,
    model,
    source_word2idx,
    target_word2idx,
    target_idx2word,
    config,
    max_summary_len=16,
    min_summary_len=5,
    temperature=0.6,
    top_k=4
):
    set_eval_mode(model)

    encoder_max_len = config.get("encoder_max_len", 120)
    decoder_max_len = min(max_summary_len, config.get("decoder_max_len", 20))

    source_pad_idx = source_word2idx.get(PAD_TOKEN, 0)
    target_pad_idx = target_word2idx.get(PAD_TOKEN, 0)
    sos_idx = target_word2idx.get(SOS_TOKEN, 1)
    eos_idx = target_word2idx.get(EOS_TOKEN, 2)
    unk_idx = source_word2idx.get(UNK_TOKEN, 3)
    target_unk_idx = target_word2idx.get(UNK_TOKEN, 3)

    text = clean_arabic_text(text)
    src_tokens = text.split()[:encoder_max_len]

    if not src_tokens:
        return ""

    src_ids = [source_word2idx.get(token, unk_idx) for token in src_tokens]
    src_ids = src_ids + [source_pad_idx] * (encoder_max_len - len(src_ids))
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    generated_words = []
    generated_ids = []
    article_tokens = set(src_tokens)

    with torch.no_grad():
        encoder_outputs, hidden = encoder_forward(
            model["encoder"],
            src_tensor,
            model["decoder"]["num_layers"]
        )

        mask = create_mask(src_tensor, source_pad_idx)
        input_token = torch.tensor([sos_idx], dtype=torch.long).to(DEVICE)

        for step in range(decoder_max_len):
            output, hidden = decoder_forward(
                model["decoder"],
                input_token,
                hidden,
                encoder_outputs,
                mask
            )

            logits = output.squeeze(0).clone()

            # Block special tokens from being generated.
            for special_idx in [target_pad_idx, sos_idx, target_unk_idx]:
                if 0 <= special_idx < logits.numel():
                    logits[special_idx] = -1e10

            # Prevent very short summaries.
            if step < min_summary_len and 0 <= eos_idx < logits.numel():
                logits[eos_idx] = -1e10

            # Repetition control.
            for token_id in generated_ids:
                if 0 <= token_id < logits.numel():
                    logits[token_id] -= 2.0

            if generated_ids and 0 <= generated_ids[-1] < logits.numel():
                logits[generated_ids[-1]] = -1e10

            # Article-aware keyword bias.
            for idx in range(logits.numel()):
                word = target_idx2word.get(idx, "")
                if word in article_tokens:
                    logits[idx] *= 1.2

            top_k = min(int(top_k), logits.numel())
            temperature = max(float(temperature), 1e-6)

            logits = logits / temperature
            topk_logits, topk_indices = torch.topk(logits, k=top_k)
            probs = torch.softmax(topk_logits, dim=-1)
            next_token = topk_indices[torch.multinomial(probs, 1)].item()

            if next_token == eos_idx:
                break

            word = target_idx2word.get(next_token, "")

            if word not in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, ""]:
                generated_words.append(word)
                generated_ids.append(next_token)

            input_token = torch.tensor([next_token], dtype=torch.long).to(DEVICE)

    return " ".join(generated_words)



# -------------------------
# Results / ROUGE helpers
# -------------------------
def load_rouge_scores(path):
    path = Path(path)
    if not path.exists():
        return None, f"ROUGE file not found: {path}"

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return None, f"Could not read ROUGE file: {e}"

    if df.empty:
        return None, "ROUGE file is empty."

    return df, None


def show_rouge_scores(path):
    rouge_df, error = load_rouge_scores(path)

    st.subheader("Focused Recall ROUGE Performance")

    if error:
        st.warning(error)
        return

    st.dataframe(rouge_df, use_container_width=True)

    metric_cols = [
        c for c in rouge_df.columns
        if c != "num_samples" and pd.api.types.is_numeric_dtype(rouge_df[c])
    ]

    if metric_cols:
        chart_df = rouge_df[metric_cols].T.reset_index()
        chart_df.columns = ["metric", "score"]
        st.bar_chart(chart_df, x="metric", y="score", use_container_width=True)

    if "num_samples" in rouge_df.columns:
        st.caption(f"Evaluated samples: {int(rouge_df['num_samples'].iloc[0])}")
    # return

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(
    page_title="Arabic Text Summarisation",
    page_icon="📝",
    layout="centered"
)

st.title("Arabic Abstractive Text Summarisation")
st.caption(f"Device: {DEVICE}")
st.write("Enter an Arabic article or sentence, and the 2-layer Seq2Seq Attention model will generate a summary.")

st.sidebar.header("Model")
st.sidebar.write("Using 2-layer model:")
st.sidebar.code(str(MODEL_PATH), language="text")

try:
    model, source_word2idx, target_word2idx, target_idx2word, config = load_model(MODEL_PATH)
    st.sidebar.success("2-layer model loaded successfully")
    st.sidebar.write("Encoder layers:", model["enc_num_layers"])
    st.sidebar.write("Decoder layers:", model["dec_num_layers"])
    st.sidebar.write("Hidden size:", model["enc_hidden_dim"])
except Exception as e:
    st.error(f"Error loading 2-layer model: {e}")
    st.stop()

st.sidebar.header("Focused Recall Decoding")
max_summary_len = st.sidebar.slider("Max summary length", 8, 30, 16, 1)
min_summary_len = st.sidebar.slider("Min summary length", 1, 12, 5, 1)
temperature = st.sidebar.slider("Temperature", 0.3, 1.2, 0.6, 0.05)
top_k = st.sidebar.slider("Top-k", 1, 15, 4, 1)

tab_generate, tab_results = st.tabs(["Generate Summary", "Performance"])

with tab_generate:
    user_text = st.text_area(
        "Enter Arabic text:",
        height=220,
        placeholder="اكتب النص العربي هنا..."
    )

    if st.button("Generate Summary"):
        if user_text.strip() == "":
            st.warning("Please enter Arabic text first.")
        else:
            with st.spinner("Generating summary..."):
                summary = summarise_text(
                    user_text,
                    model,
                    source_word2idx,
                    target_word2idx,
                    target_idx2word,
                    config,
                    max_summary_len=max_summary_len,
                    min_summary_len=min_summary_len,
                    temperature=temperature,
                    top_k=top_k
                )

            st.subheader("Generated Summary")
            if summary.strip():
                st.success(summary)
            else:
                st.error("The model could not generate a summary for this input.")

with tab_results:
    show_rouge_scores(ROUGE_PATH)

    loss_curve_path = RESULTS_DIR / "loss_curve.png"
    if loss_curve_path.exists():
        st.subheader("Training vs Validation Loss")
        st.image(str(loss_curve_path), use_container_width=True)
