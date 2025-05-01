# 📱 SMS Spam Classifier

A machine learning web app that classifies SMS messages as **Spam** or **Not Spam**. Trained on a dataset of 5,573 labeled messages, this tool uses natural language processing (NLP) techniques and a classification model to identify unwanted or harmful messages.

---

## 📊 Dataset

- **Source**: [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data)
- **Total Samples**: 5,573 messages
- **Classes**:
  - `ham`: Not spam
  - `spam`: Unwanted message

---

## 🧠 Model & Techniques

- **Text Preprocessing**:
  - Lowercasing
  - Removing stopwords & punctuation
  - Tokenization
 

- **Feature Extraction**:
  - TF-IDF Vectorization

- **Model**:
  - Multinomial Naive Bayes / Logistic Regression/Support Vector Classifier (choose one)

- **Accuracy** 0.9816247582205029
- **Precision** 0.9917355371900827

---

## 🖥️ Web App (with Streamlit)

Built using **Streamlit** to provide an interactive and user-friendly interface.

### 🔧 Requirements

```bash
pip install -r requirements.txt
streamlit run app.py

```

---

✨ Features
 Instant classification of SMS messages
 Displays prediction result: ✅ Not Spam or 🚫 Spam
 Clean and responsive web UI

 ---


 🔗 Live Demo

![image](https://github.com/user-attachments/assets/f1841736-0cc8-46f7-bc31-0d7e543c75af)
![image](https://github.com/user-attachments/assets/e8ece0a5-8a53-4710-9cbc-5186f9a1e7a5)


👉 [Click here to try the app](https://spamclassifier-1.streamlit.app/)
