# Arabic Abstractive Text Summarisation using Seq2Seq with Attention

## Overview

This project implements an **Arabic Abstractive Text Summarisation system** using **Sequence-to-Sequence (Seq2Seq) neural networks with an attention mechanism**.
The goal is to generate concise summaries from long Arabic news articles while preserving the key information and meaning of the original text.

The project includes a full NLP pipeline:

* Arabic text preprocessing
* Dataset preparation
* Seq2Seq model training
* Evaluation using ROUGE metrics
* A simple interface for generating summaries.

---

## Datasets

The model is trained and evaluated using multiple Arabic summarisation datasets:

* **SumArabic**
* **AraSum**
* **Arabic Text Summarisation Dataset**
* **Egyptian Arabic Summarisation Dataset**

These datasets contain **article–summary pairs** that allow the model to learn how to generate summaries.

Example dataset entry:

```
Article:
أعلنت وزارة الصحة اليوم عن إطلاق حملة وطنية جديدة للتطعيم ضد الإنفلونزا الموسمية، وذلك بهدف تقليل انتشار المرض خلال فصل الشتاء. وأكدت الوزارة أن اللقاح متوفر في جميع المراكز الصحية الحكومية، ودعت المواطنين إلى المبادرة بالحصول عليه خاصة كبار السن والأشخاص الذين يعانون من أمراض مزمنة.
Summary:
وزارة الصحة تطلق حملة وطنية للتطعيم ضد الإنفلونزا
```

---

## Project Structure

```
arabic-text-summarisation-seq2seq
│
├── notebooks
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_dataset_analysis.ipynb
│   └── 03_model_training.ipynb
│
├── src
│   ├── preprocessing.py
│   ├── tokeniser.py
│   ├── model.py
│   └── training.py
│
├── data
|   ├── raw_data
│   └── processed_data
│
├── app
│   └── streamlit_app.py
│
├── results
│   ├── rouge_scores.csv
│   └── training_plots
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Data Preprocessing

Arabic text requires specialised preprocessing before training neural models.

The preprocessing pipeline includes:

* Removing URLs and special characters
* Arabic normalisation
* Removing diacritics (Tashkeel)
* Tokenisation
* Cleaning duplicated or empty samples

Example normalisation:

```
إقتصاد → اقتصاد
الْكِتَاب → الكتاب
```

---

## Model Architecture

The summarisation system is based on a **Seq2Seq architecture with attention**.

### Encoder

* Bidirectional LSTM / GRU
* Encodes the input article

### Decoder

* LSTM / GRU
* Generates the summary word-by-word

### Attention Mechanism

The attention layer helps the model focus on the most important parts of the input text when generating each word in the summary.

---

## Training

Training is performed on the **training split** of the datasets with validation monitoring.

Typical training steps:

1. Load datasets
2. Apply preprocessing
3. Tokenise text
4. Convert to numerical sequences
5. Train Seq2Seq model
6. Evaluate performance

---

## Evaluation

The model is evaluated using **ROUGE metrics**, which measure the overlap between generated summaries and reference summaries.

Metrics used:

* ROUGE-1
* ROUGE-2
* ROUGE-L

Higher scores indicate better summarisation quality.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/arabic-text-summarisation-seq2seq.git
cd arabic-text-summarisation-seq2seq
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Run preprocessing notebook

```
notebooks/01_data_preprocessing.ipynb
```

### Train the model

```
notebooks/03_model_training.ipynb
```

### Run the Streamlit interface

```bash
streamlit run app/streamlit_app.py
```

---

## Future Improvements

Possible improvements include:

* Transformer-based models (BERT, T5, BART)
* Better Arabic tokenisation (Farasa / CAMeL Tools)
* Larger datasets
* Reinforcement learning for summarisation
* Fine-tuning pretrained Arabic language models

---

## Technologies Used

* Python
* Pandas
* PyTorch / TensorFlow
* NumPy
* Streamlit
* Matplotlib

---

## Author

NLP Final Project
Arabic Text Summarisation System

---

## License

This project is for **academic and research purposes**.

---

## Contributors

This project was developed as part of a **Natural Language Processing course project**.

* Mohammad O. ALADDASI
* Hamza Rashdan
* Saif Sharkasi 

Supervisor: Instructor Name



