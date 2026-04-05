# Module 1 - Sentiment in Text: Tokenization, Padding, and NLP Preprocessing

> **Turning raw text into model-ready sequences with TensorFlow/Keras.**  
> This module focuses on the foundational preprocessing pipeline behind modern NLP systems: tokenization, vocabulary building, out-of-vocabulary handling, sequence padding, stopword filtering, and label encoding.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/) [![Keras](https://img.shields.io/badge/Keras-Tokenizer-red.svg)](https://keras.io/) [![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)

---

## Overview

In computer vision, models consume pixels. In NLP, models consume numbers.  
That means raw text must first be converted into a structured numerical representation before any sentiment model or text classifier can learn from it.

In this module, I built the preprocessing foundation for NLP tasks using TensorFlow and Keras. Starting from simple example sentences and progressing to real-world text datasets, I implemented a pipeline that:

- converts text into tokens
- creates vocabulary indexes
- handles unseen words with `<OOV>`
- pads variable-length sequences
- removes stopwords
- transforms categorical labels into numeric sequences

This module is the bridge between **plain language** and **deep learning-ready tensors**.

---

## Why This Matters

Any real NLP system — sentiment analysis, topic classification, toxicity detection, chatbot intent routing, spam detection, or sarcasm detection — depends on solid text preprocessing.

This project demonstrates the exact skills required to prepare unstructured text for machine learning:

- cleaning noisy language
- standardizing variable-length inputs
- building reproducible preprocessing pipelines
- preparing both features and labels for model training

These are practical, portfolio-worthy skills for roles in:

- Machine Learning Engineering
- Data Science
- NLP Engineering
- Applied AI

---

## Learning Objectives

In this module, I learned to:

- tokenize raw text using Keras `Tokenizer`
- build a word index from text corpora
- manage unseen words with `oov_token`
- convert sentences into integer sequences
- apply padding to normalize variable-length text
- preprocess real-world datasets stored in JSON and CSV formats
- remove stopwords from news text
- encode text labels into machine-readable numeric sequences

---

## Skills Demonstrated

### 1. NLP Preprocessing
- Tokenization of natural language text
- Vocabulary creation and word-to-index mapping
- Sequence generation from sentences
- Out-of-vocabulary token handling

### 2. Sequence Engineering
- Padding shorter sequences to a common length
- Preserving consistent tensor shapes for downstream deep learning models
- Comparing fixed-length vs variable-length text representations

### 3. Real-World Dataset Handling
- Reading structured text data from JSON and CSV files
- Extracting text, labels, and metadata fields
- Cleaning text with stopword filtering before tokenization

### 4. Label Encoding
- Transforming categorical text labels into integer sequences
- Preparing labels for supervised text classification tasks

---

## What’s Inside This Module

### Lesson 1 — Tokenizer Fundamentals
A simple introduction to how Keras tokenizes text and builds a vocabulary from sentences.

**Key idea:**  
Different words are mapped into integer IDs so they can be consumed by neural networks.

---

### Lesson 2 — OOV Tokens and Padding
This lesson introduces two critical NLP concepts:

- **`<OOV>` tokens** for words not seen during training
- **`pad_sequences`** to make variable-length text uniform

**Key idea:**  
Real text is messy and inconsistent in length. Models need standardized inputs.

---

### Lesson 3 — Real-World Text Pipeline with Sarcasm Headlines
This notebook moves from toy examples to a real JSON dataset of news headlines and sarcasm labels.

**Key idea:**  
The same tokenizer/padding workflow can scale from tiny examples to real NLP datasets.

---

### Exercise — BBC News Text Preprocessing
This exercise applies the full preprocessing pipeline to a BBC text dataset in CSV format.

It includes:

- reading article text and category labels
- removing stopwords
- building a vocabulary with Keras `Tokenizer`
- converting text to sequences
- padding sequences
- encoding labels into integer sequences

**Exercise output highlights:**
- **2,225** text samples
- **~29.7k** token vocabulary
- padded sequence shape of **(2225, 2442)**
- **5 encoded label classes**

This is a strong example of transforming raw text into model-ready inputs for downstream classification.

---

## Technical Highlights

### Tokenization

```python
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
```

### Sequence Conversion

```python
sequences = tokenizer.texts_to_sequences(sentences)
```

### Padding

```python
padded = pad_sequences(sequences, padding='post')
```

###Label Enconding

```python
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
label_seq = label_tokenizer.texts_to_sequences(labels)
```

---

## Datasets Used

### 1. Toy Sentences

Used to understant how tokenization works at a conceptual level.

### 2. Sarcasm Headlines Dataset

A *JSON* dataset used for text preprocessing and category encoding.

This progression from small controlled examples to real-world corpora makes the module especially valuable from a learning and portfolio perspective.

---

## Key Concepts

### 1. Tokenization

Tokenization is the process of breaking text into smaller units — usually words — and assigning each one a numeric ID.

### 2. Vocabulary

A vocabulary is the set of unique words recognized by the tokenizer.

### 3. Out-of-Vocabulary (OOV) Handling

When the model encounters a word it has never seen before, the <OOV> token prevents the pipeline from breaking.

### 4. Padding

Neural networks require consistent input shapes. Padding ensures all sequences have the same length.

### 5. Stopword Removal

Common words such as “the”, “and”, and “is” may add noise depending on the task. Removing them can help highlight more informative terms.

### 6. Label Encoding

Text labels must also be converted to numbers so the model can learn from them.

---

## Why Recruiters Should Care

This module shows more than coursework completion. It demonstrates that I understand a core production NLP workflow:

- Ingest raw text data
- Clean and standardize it
- Convert it to a machine-readable representation
- Prepare both features and labels for supervised learning

That is the groundwork behind systems used in:

- Sentiment analysis
- Support ticket classification
- Social media monitoring
- Headline categorization
- Intent detection
- Recommendation and moderation pipelines

---

## Folder structure

Module1_Sentiment_in_Text
├── Course_3_Week_1_Lesson_1.ipynb
├── Course_3_Week_1_Lesson_2.ipynb
├── Course_3_Week_1_Lesson_3.ipynb
├── Course_3_Week_1_Exercise_question.ipynb
├── Course_3_Week_1_Exercise_solution.ipynb
└── README.md

---

## ▶️ How to Run

This module consists of Jupyter notebooks that can be run locally or on Google Colab.

### Prerequisites

- Python 3.8 or higher
- pip
- Virtual environment support (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/victorperone/Natural_Language_Preprocessing_in_Tensorflow.git
cd Module1_Text_Sentiment_Analysis
```

### 2. Create and Activate a Virtual Environment (Recommended)

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

All required packages are listed in `requirements.txt`

```python
pip install -r requirements.txt
```
This will install TensorFlow and other necessary libraries.

### 4. Launch Jupyter Notebook

You can run the notebooks **locally** or using **Google Colab**.

#### Option A: Run Locally (Jupyter Notebook)

```bash
jupyter notebook
```

or, if you prefer JupyterLab:
```bash
jupyter lab
```

### Option B: Run on Google Colab (No Local Setup Required)

1. Go to: [Google Colab](https://colab.research.google.com)
2. Click File → Open notebook
3. Select the GitHub tab
4. Paste your repository URL
5. Open  `File.ipynb`

Google Colab provides:

- Free CPU (and optional GPU) execution
- No local Python or TensorFlow installation
- Automatic dependency handling for most libraries

⚠️ Note: If requirements.txt is not automatically handled, install dependencies in a Colab cell:

```python
!pip install -r requirements.txt
```

### 5. Run the Exercises

Open the notebooks in numerical order
Run each cell sequentially
Observe how changes in model architecture, training duration, and callbacks affect results
It is recommended to run the exercises **in order**, as each one builds conceptually on the previous examples.

### Environment Notes

- TensorFlow may produce informational or warning messages during execution.
- These messages do not affect the correctness of the exercises.
- CPU execution is sufficient for all notebooks in this module, though GPU is recommended for faster training on the 300x300 images.

---

## 🧪 Reproducibility Note

Model training involves random initialization of weights.
As a result:
- Exact accuracy and loss values may vary slightly between runs
- Overall trends and conclusions should remain consistent
- Pretrained weights ensure consistent feature extraction 
- Minor accuracy variations may occur due to:
  - Random initialization of dense layers

---

## Results

By the end of this module, I successfully built a complete preprocessing workflow for NLP tasks using TensorFlow/Keras.

I transformed raw text into padded numerical sequences and encoded labels into machine-readable form, creating the exact kind of structured input required for later sentiment analysis and text classification models.

---

## Next Step

This module establishes the preprocessing layer for later NLP architectures such as:

Embedding layers
Dense neural networks for text classification
RNNs / LSTMs
Sequence models for sentiment and semantic tasks

---

## Summary

This project demonstrates practical NLP preprocessing with TensorFlow and Keras, covering tokenization, OOV handling, padding, stopword removal, and label encoding on both toy and real-world datasets.

It is a foundational step toward building end-to-end sentiment analysis and text classification systems.

















