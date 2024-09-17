# import streamlit as st
# import pickle
# import string
# import nltk
# from  nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer

# nltk.download('punkt')
# nltk.download('stopwords')

# ps = PorterStemmer()

# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)

# tfidf = pickle.load(open('vectorizer1.pkl','rb'))
# model = pickle.load(open('model1.pkl','rb'))

# st.title("Email/Sms spam classifier")
# input_sms = st.text_input("Enter the message")

# if st.button('Predict'):

#     # 1. preprocess
#     transformed_sms = transform_text(input_sms)
#     # 2. vectorize
#     vector_input = tfidf.transform([transformed_sms])
#     # 3. predict
#     result = model.predict(vector_input)[0]
#     # 4. Display
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

st.set_page_config(
    page_title="Spam Classifier"
)

def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize

    # Remove non-alphanumeric characters
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuations
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    # Apply stemming
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer1.pkl', 'rb'))
model = pickle.load(open('model1.pkl', 'rb'))

# --- Theme Toggle ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'  # Set default theme to light

# Toggle button to change theme
theme = st.sidebar.radio("Select Theme", ['Light Mode', 'Dark Mode'])

if theme == 'Light Mode':
    st.session_state.theme = 'light'
else:
    st.session_state.theme = 'dark'

# --- CSS for Themes ---
if st.session_state.theme == 'light':
    theme_css = '''
    <style>
    body {
        background-color: white;
        color: #000000;
    }
    .stButton > button {
        background-color: white;
        color: black;
        text:bold
    }
    .stTextArea {
        border-color: #c3c3c3;
        border-radius: 4px;
    }
    </style>
    '''
else:
    theme_css = '''
    <style>
    body {
        background-color: #333333;
        color: #ffffff;
    }
    .stButton > button {
        background-color: #444444;
        color: #ffffff;
    }
    .stTextArea {
        border-color: #555555;
        border-radius: 4px;
        background-color: #444444;
        color: #ffffff;
    }
    </style>
    '''

st.markdown(theme_css, unsafe_allow_html=True)

# --- Main App Layout ---
st.markdown("<h1 style='text-align: center; color: bisque;'>📧 Spam Classifier</h1>", unsafe_allow_html=True)

# Initialize session state for input_sms if not already set
if 'input_sms' not in st.session_state:
    st.session_state.input_sms = ""

# Add columns for better organization
col1, col2 = st.columns([2, 1])

with col1:
    # Input field for SMS/Email message
    input_sms = st.text_area("Enter the message:", st.session_state.input_sms, placeholder="Type your email or SMS here...")

    # Show a button to display an example
    # if st.button("Show Example"):
    #     st.session_state.input_sms = "Congratulations! You've won a $1000 gift card. Click here to claim now!"

    # Predict button
    predict_button = st.button("Predict")

with col2:
    # Display loading spinner when processing
    if predict_button:
        with st.spinner('Classifying message...'):
            if input_sms.strip():
                # Preprocess the input message
                transformed_sms = transform_text(input_sms)

                # Vectorize the input message
                vector_input = tfidf.transform([transformed_sms])

                # Predict whether it's spam or not
                result = model.predict(vector_input)[0]

                # Display the result
                if result == 1:
                    st.error("🚨 This message is **Spam**!")
                else:
                    st.success("✅ This message is **Not Spam**!")
            else:
                st.warning("⚠️ Please enter a valid message for classification.")
    
    # Option to reset input
    if st.button('Clear Message'):
        st.session_state.input_sms = ""  # Reset the session state variable

# Footer or additional information
st.markdown("---")
st.markdown("#### 📝 Notes:")
st.markdown("""
- **Accuracy:** This model provides a basic spam classification using a machine learning algorithm.
- **Data Privacy:** All text you input here is processed on your local machine; no data is sent to a server.
- **Preprocessing:** The input message is converted to lowercase, tokenized, stripped of stopwords and punctuation, and then stemmed before classification.
""")
