import streamlit as st
from openpyxl import Workbook
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sample training data for text classification
training_data = [
    ("I was not able to find the OA", "missing_oa"),
    ("I was not able to find the material","missing_material"),
    ("I was not able to find the OA and material",'missing_oa_and_material'),
    ("Price changes more than 50%",'big_changes'),
    ("I'm not sure how to respond to that. Can you please provide more details?", "unclear_request"),
]

# Extract features and labels
X_train, y_train = zip(*training_data)

# Create a text classification pipeline
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    ('clf', MultinomialNB())
])

# Train the text classification model
text_clf.fit(X_train, y_train)

# New warning response generator function
def warning_response_generator(user_input):
    if "oa" in user_input.lower():
        return "I was not able to find the OA"
    elif 'material' in user_input.lower():
        return "I was not able to find the material"
    elif "oa" in user_input.lower() and 'material' in user_input.lower():
        return "I was not able to find the OA and material"
    elif 'double' in user_input.lower():
        return "Price changes more than 50%"
    else:
        return "I'm not sure how to respond to that. Can you please provide more details?"

# Function to classify warning responses
def classify_text(warning_response):
    prediction = text_clf.predict([warning_response])[0]
    return prediction

# Function to save chat history to Excel
def save_to_excel(messages):
    wb = Workbook()
    ws = wb.active
    ws.append(["User Input", "Bot Response"])

    for message in messages:
        ws.append([message["User"], message["Bot"]])

    excel_bytes = BytesIO()
    wb.save(excel_bytes)
    excel_bytes.seek(0)

    return excel_bytes

# Function to save classified messages to Excel
def save_classification_to_excel(messages):
    wb = Workbook()
    ws = wb.active
    ws.append(["User Input", "Warning Response", "Classification"])

    for message in messages:
        warning_response = warning_response_generator(message['User'])
        classification = classify_text(warning_response)
        ws.append([message["User"], warning_response, classification])

    excel_bytes = BytesIO()
    wb.save(excel_bytes)
    excel_bytes.seek(0)

    return excel_bytes

# Display the chat history
for message in st.session_state.messages:
    st.markdown(f"**User:** {message['User']}")
    st.markdown(f"**Bot:** {message['Bot']}")
    st.write("---")

# Input form
with st.form(key="input_form"):
    user_input = st.text_input(" ", key="user_input", placeholder="Describe your Outline Agreement update request")
    submit_button = st.form_submit_button(label="Send")

if submit_button and user_input:
    warning_response = warning_response_generator(user_input)
    classification = classify_text(warning_response)
    st.session_state.messages.append({"User": user_input, "Bot": warning_response, "Warning": classification})
    st.rerun()

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("Report"):
        excel_bytes = save_to_excel(st.session_state.messages)
        st.download_button(
            label="Download Excel",
            data=excel_bytes,
            file_name="chat_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

with col2:
    if st.button("Download Classification"):
        excel_bytes = save_classification_to_excel(st.session_state.messages)
        st.download_button(
            label="Download Classification Excel",
            data=excel_bytes,
            file_name="classification_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
