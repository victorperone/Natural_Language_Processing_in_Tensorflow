# Natural_Language_Processing_in_Tensorflow

> **From Text Preprocessing to Language Generation**: building practical NLP systems with TensorFlow, from sentiment analysis and word embeddings to sequence models and literary text generation.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/) [![Keras](https://img.shields.io/badge/Keras-NLP-red.svg)](https://keras.io/) [![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/) ![Status](https://img.shields.io/badge/Status-Complete-success.svg)

This repository documents my implementation and analysis of the models, exercises, and experiments developed throughout the course **Natural Language Processing in TensorFlow**.

It covers the full progression of practical NLP with TensorFlow:

- Text preprocessing
- Sentiment analysis
- Trainable word embeddings
- Recurrent and convolutional sequence models
- Next-word prediction
- Literary text generation

🔗 **Course link:**  

[Natural Language Processing in TensorFlow](https://www.coursera.org/learn/natural-language-processing-tensorflow)

---

## 🎯 Focus of This Repository

Rather than simply storing course notebooks, this repository emphasizes:

- Clear module-by-module documentation
- Practical NLP workflow understanding
- Architectural progression from preprocessing to generation
- Readable organization with dedicated READMEs and supporting assets

---

## 📘 Course Overview

This repository corresponds to the **third course in the TensorFlow specialization**, focused on Natural Language Processing.

The course progresses through four major stages:

1. **Text preprocessing and sentiment analysis**
2. **Word embeddings and semantic representation**
3. **Sequence models for contextual understanding**
4. **Language generation with literary corpora**

Together, these modules build a strong practical foundation in NLP engineering with TensorFlow and Keras.

### **Course Modules**

1. **Text Sentiment Analysis**  
   Focuses on preprocessing foundations such as tokenization, padding, stopword removal, and label encoding, using real datasets to prepare text for machine learning.

2. **Word Embedding**  
   Introduces trainable word vectors, TensorFlow Datasets, subword tokenization, and embedding visualization, moving from raw token IDs to meaningful semantic representations.

3. **Sequence Models**  
   Explores Bidirectional LSTM, stacked recurrent models, GRU, Conv1D, and hybrid architectures, showing how sequence models capture context, order, and temporal dependencies in language.

4. **Sequence Models and Literature**  
   Shifts from sequence understanding to language generation through next-word prediction, autoregressive generation loops, and literary text modeling on lyrics and sonnets.

---

## 💡 Why This Matters

Practical NLP skills like these are foundational for building systems such as:

- Sentiment analysis pipelines
- Document and topic classifiers
- Intent detection systems
- Recommendation and semantic search tools
- Chatbot and assistant interfaces
- Language generation workflows

This repository shows a full learning path from **structured text preprocessing** to **generative language modeling**, which is one of the most valuable progressions in modern NLP learning.

---

## 🛠️ Key Concepts Mastered

| Concept | Description |
| :--- | :--- |
| **Text Preprocessing** | Cleaned and prepared raw text using tokenization, padding, stopword removal, OOV handling, and label encoding. |
| **Word Embeddings** | Learned dense vector representations of words and explored how embeddings capture semantic similarity. |
| **Sequence Modeling** | Built Bidirectional LSTM, GRU, stacked recurrent, and Conv1D architectures for contextual text understanding. |
| **Subword Tokenization** | Applied subword-based encodings to make models more robust to rare and unseen words. |
| **Language Modeling** | Framed next-word prediction as supervised learning and trained models to estimate token probabilities over a vocabulary. |
| **Autoregressive Generation** | Generated new text iteratively by feeding model predictions back into the next input context. |

---

## 📊 Results & Highlights

- Built an end-to-end NLP portfolio that progresses from preprocessing to generation
- Worked with diverse datasets including IMDB reviews, BBC News, sarcasm headlines, Irish lyrics, and sonnets
- Implemented multiple text-modeling architectures with TensorFlow and Keras
- Explored both classification tasks and generative language modeling
- Created module-level documentation and SVG diagrams to present the work more clearly


> **Together, these projects demonstrate practical NLP engineering across preprocessing, representation learning, contextual sequence modeling, and text generation.**
---

## 🧠 Engineering Lessons Learned

- Preprocessing choices strongly influence downstream model behavior
- Embeddings are the bridge between raw token IDs and semantic learning
- Sequence models are essential when context and word order matter
- Architecture choice should match the task, whether classification or generation
- Next-word prediction is a powerful conceptual foundation for modern generative AI
- Clear project documentation is part of good machine learning engineering

---

## 🔭 How This Prepares Me for Real-World ML / NLP

This course strengthened my ability to:

- Design NLP pipelines from raw text to model-ready features
- Work with binary, multiclass, and generative sequence tasks
- Select appropriate model families for different NLP problems
- Explain architecture tradeoffs clearly and professionally
- Organize technical work into a portfolio that is easy to review

---

## 📁 Repository Content

This repository includes:

- Jupyter notebooks for each module
- Module-level README files with explanations and results
- Architecture diagrams and visual summaries
- Requirements files for reproducibility
- Course exercises and supporting assets

---

## 📂 Folder Structure

<pre>
📦 Natural_Language_Processing_in_Tensorflow
├── 📁 <a href="https://github.com/victorperone/Natural_Language_Processing_in_Tensorflow/tree/main/Module1_Text_Sentiment_Analysis">Module1_Text_Sentiment_Analysis</a>
│   ├── 📓 Course_3_Week_1_Lesson_1.ipynb
│   ├── 📓 Course_3_Week_1_Lesson_2.ipynb
│   ├── 📓 Course_3_Week_1_Lesson_3.ipynb
│   ├── 📓 Course_3_Week_1_Exercise_question.ipynb
│   ├── 📓 Course_3_Week_1_Exercise_solution.ipynb
│   ├── 📁 architectures
│   │   └── 🏗️ module1_nlp_pipeline_polished.svg
│   ├── 📄 requirements.txt
│   └── 📘 README.md
├── 📁 <a href="https://github.com/victorperone/Natural_Language_Processing_in_Tensorflow/tree/main/Module2_Word_Embedding">Module2_Word_Embedding</a>
│   ├── 📓 Course_3_Week_2_Lesson_1.ipynb
│   ├── 📓 Course_3_Week_2_Lesson_2.ipynb
│   ├── 📓 Course_3_Week_2_Lesson_3.ipynb
│   ├── 📓 Course_3_Week_2_Exercise_Question.ipynb
│   ├── 📁 architectures
│   │   ├── 🏗️ imdb_embedding_model.svg
│   │   ├── 🏗️ sarcasm_embedding_model.svg
│   │   ├── 🏗️ subword_embedding_model.svg
│   │   └── 🏗️ bbc_multiclass_embedding_model.svg
│   ├── 📄 requirements.txt
│   └── 📘 README.md
├── 📁 <a href="https://github.com/victorperone/Natural_Language_Processing_in_Tensorflow/tree/main/Module3_Sequence_models">Module3_Sequence_models</a>
│   ├── 📓 Course_3_Week_3_Lesson_1a_IMDB Subwords 8K with Single Layer LSTM.ipynb
│   ├── 📓 Course_3_Week_3_Lesson_1b_IMDB Subwords 8K with Multi Layer LSTM.ipynb
│   ├── 📓 Course_3_Week_3_Lesson_1c_IMDB Subwords 8K with 1D Convolutional Layer.ipynb
│   ├── 📓 Course_3_Week_3_Lesson_2_Sarcasm with Bidirectional LSTM.ipynb
│   ├── 📓 Course_3_Week_3_Lesson_2c_Sarcasm with 1D Convolutional Layer.ipynb
│   ├── 📓 Course_3_Week_3_Lesson_2d_IMDB Reviews with GRU (and optional LSTM and Conv1D).ipynb
│   ├── 📓 Activity_C3_W3_Assignment.ipynb
│   ├── 📁 architectures
│   │   ├── 🏗️ lstm_explained.svg
│   │   ├── 🏗️ gru_explained.svg
│   │   ├── 🏗️ single_vs_multi_layer.svg
│   │   ├── 🏗️ unidirectional_vs_bidirectional.svg
│   │   ├── 🏗️ module3_single_layer_bilstm.svg
│   │   ├── 🏗️ module3_multi_layer_bilstm.svg
│   │   ├── 🏗️ module3_conv1d_sequence_model.svg
│   │   ├── 🏗️ module3_sarcasm_bidirectional_lstm.svg
│   │   ├── 🏗️ module3_sarcasm_conv1d.svg
│   │   ├── 🏗️ module3_bidirectional_gru.svg
│   │   └── 🏗️ module3_hybrid_glove_conv_lstm.svg
│   ├── 📄 requirements.txt
│   └── 📘 README.md
├── 📁 <a href="https://github.com/victorperone/Natural_Language_Processing_in_Tensorflow/tree/main/Module4_Sequence_Models_and_Literature">Module4_Sequence_Models_and_Literature</a>
│   ├── 📓 C3_W4_Lab_1.ipynb
│   ├── 📓 C3_W4_Lab_2_irish_lyrics.ipynb
│   ├── 📓 C3_W4_Exercício.ipynb
│   ├── 📓 C3_W4_Exercício_2.ipynb
│   ├── 📁 architectures
│   │   ├── 🏗️ module4_language_generation_pipeline_fixed.svg
│   │   ├── 🏗️ module4_autoregressive_loop.svg
│   │   ├── 🏗️ module4_baseline_generator.svg
│   │   ├── 🏗️ module4_irish_lyrics_generator.svg
│   │   └── 🏗️ module4_deep_sonnet_generator.svg
│   ├── 📄 requirements.txt
│   └── 📘 README.md
├── 📄 requirements.txt
└── 📘 README.md

</pre>

Legend:

<pre>
📁 Folder
📓 Jupyter Notebook
🏗️ Model Architecture / Diagram (.svg)
📊 Results / Plots (.png) 
🗜️ Compressed Dataset 
📄 Configuration File
📘 Documentation
</pre>

---

## 🚀 Getting Started

To run the notebooks:

```bash
# Clone the repository
git clone https://github.com/victorperone/Natural_Language_Processing_in_Tensorflow.git
cd Natural_Language_Processing_in_Tensorflow

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

If you prefer to work module by module, each folder also includes its own `requirements.txt` and `README.md`.

---

## 🛠️ Technologies Used:

- TensorFlow 2.x (Keras API)
- TensorFlow Datasets
- NumPy
- Matplotlib
- JSON / CSV text processing
- Google Colab / Jupyter Notebook

---

## 📚 References

- TensorFlow Documentation: https://www.tensorflow.org
- Coursera: Natural Language Processing in TensorFlow: https://www.coursera.org/learn/natural-language-processing-tensorflow
