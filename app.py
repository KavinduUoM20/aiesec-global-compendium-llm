import streamlit as st
from streamlit_chat import message
from utils import create_retrieval
import os

# Load OpenAI API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Headlines
st.title("AIESEC In Sri Lanka")
st.subheader("Chat with Global Compendium (β Version)")

# Sessions
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hi! Welcome to AIESEC In Sri Lanka.\nHow can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Container for chat history
response_container = st.container()

# Container for text box
textcontainer = st.container()

with textcontainer:
    query = st.chat_input("Enter Your Question..", key="input")

    if query:
        with st.spinner("typing..."):
            chain = create_retrieval()
            response = chain.invoke({"input": query})
            response = response["answer"]
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)
        st.session_state["query"] = ""

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')





st.write("<br><br><br>", unsafe_allow_html=True)
#Footer
st.write("<p style='text-align: center;'>Made with ❤️ by &lt;/Dev.Team&gt; of AIESEC in Sri Lanka</p>", unsafe_allow_html=True)