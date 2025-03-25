import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from bhashini_services1 import Bhashini_master
from audio_recorder_streamlit import audio_recorder
from PIL import Image
import base64
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings    
st.set_page_config(page_title="सेवा सहायक", page_icon="🤖", layout="wide")

# Add background image from a local file
def add_bg_from_local(image_file, opacity=0):
    with open(image_file, "rb") as image:
        encoded_image = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(255, 255, 255, {opacity}), rgba(255, 255, 255, {opacity})),
                              url(data:image/jpg;base64,{encoded_image});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            height: 100vh;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the background function (update the image path as needed)
add_bg_from_local('image/grey_bg.jfif')

# Custom CSS for header, table, and footer styling
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #002F74;
            color: white;
            text-align: center;
            padding: 5px;
            font-weight: bold;
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
        }
        .footer p {
            font-style: italic;
            font-size: 14px;
            margin: 0;
            flex: 1 1 50%;
        }
        .title {
            margin-bottom: 30px;
            word-wrap: break-word;
        }
        .dataframe td {
            max-width: 600px;
            white-space: normal;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .dataframe th {
            text-align: left;
        }
        .dataframe tr:hover {
            background-color: #f1f1f1;
        }
        .columns-wrapper {
            margin-left: 30px;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="columns-wrapper">', unsafe_allow_html=True)

# Header: Display logos and title using a three-column container
with st.container():
    col1, col2, col3 = st.columns([0.1, 0.6, 0.1], gap="small")
    with col1:
        st.markdown(
            '<div style="text-align: left; margin-bottom: -100px; display: flex; flex-direction: column; justify-content: flex-start;">',
            unsafe_allow_html=True
        )
        logo_image = Image.open('image/public-icon.jpg')
        st.image(logo_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
         st.markdown(
            """
            <div style="text-align: center; margin-left: 30px;">
                <h1 style="color:#000080; margin-bottom: 0;">सेवा सहायक 🤖</h1>
                <p style="font-size: 18px; font-weight: 600; margin-top: 5px;">AI-based chatbot for citizen services</p>
            </div>
            """,
            unsafe_allow_html=True
         )
    with col3:
        gov_logo = Image.open('image/mpsedc-logo.png')
        gov_logo_resized = gov_logo.resize((165, 127))
        st.image(gov_logo_resized)
st.markdown('</div>', unsafe_allow_html=True)

# Introductory text
st.markdown(
    """<div class="title">Welcome to the सेवा सहायक!</div>""",
    unsafe_allow_html=True
)
# st.markdown("This tool allows you to interact with our citizen services through a conversational AI. You can either speak or type your query.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

# Load environment variables
load_dotenv()

# st.title("Citizen Service Chatbot")

# Available languages for selection
languages = {
    "English": "en",
    "Hindi": "hi",
    # "Bengali": "bn",
    # "Punjabi": "pa",
    # "Telugu": "te",
}
selected_language = st.selectbox("Query language:", options=list(languages.keys()))
language_code = languages[selected_language]
# st.write(f"Selected language: {selected_language}")
fallback_language_code = languages[selected_language]

api_key = st.secrets["secret_section"]["openai_api_key"]
bhashini_url = st.secrets["secret_section"]["bhashini_url"]
bhashini_authorization_key = st.secrets["secret_section"]["bhashini_authorization_key"]
bhashini_ulca_api_key = st.secrets["secret_section"]["bhashini_ulca_api_key"]
bhashini_ulca_userid = st.secrets["secret_section"]["bhashini_ulca_userid"]

# Initialize Bhashini master for transcription
bhashini_master = Bhashini_master(
    url=bhashini_url,
    authorization_key=bhashini_authorization_key,
    ulca_api_key=bhashini_ulca_api_key,
    ulca_userid=bhashini_ulca_userid
)

# Directory for FAISS index
PERSIST_DIR = os.path.join(os.getcwd(), "faiss_index_eoffice")
if not os.path.exists(PERSIST_DIR):
    print("❌ FAISS index not found! Rebuild it first.")
def load_faiss_vectorstore():
    """Load FAISS vector store from disk and verify dimensions."""
    if not os.path.exists(PERSIST_DIR):
        st.error("FAISS index not found. Please rebuild the FAISS index using the correct embedding model.")
        return None
    model_name = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    # embeddings = OpenAIEmbeddings(api_key=api_key)
    expected_dim = len(embeddings.embed_query("test query"))
    
    try:
        vector_store = FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)
        if vector_store.index.d != expected_dim:
            st.error(f"Dimension mismatch: expected {expected_dim}, but index has {vector_store.index.d}. Please rebuild the FAISS index.")
            return None
        # st.success("FAISS index loaded successfully!")
        return vector_store
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        return None

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(model="gpt-4", api_key=api_key, temperature=0.3)
    # Use similarity search instead of mmr
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    # st.write("Testing Retriever: Fetching Context...")
    test_query = "What services are available for citizens?"
    retrieved_docs = retriever.get_relevant_documents(test_query)
    # st.write(f"Retrieved {len(retrieved_docs)} documents for test query.")
    
    prompt_template = """You are a knowledgeable and helpful assistant with expertise in the subject matter.

Context:
{context}

User’s Question:
"{question}"

Instructions:
Please provide a detailed and well-organized answer that directly addresses the question. Use the context above to support your answer, and if you encounter any gaps in your knowledge, suggest reliable sources or ask for further clarification. Include examples or references where appropriate, and clearly explain your reasoning.
"""
    
    # Removed "language" from input_variables since it's not used in the template
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"],
            ),
        }
    )
    
    return qa_chain

def get_response(user_input):
    # st.write(f"Debug: Received query - {user_input}")
    vector_store = load_faiss_vectorstore()
    if not vector_store:
        st.error("Vector store not found. Please rebuild the FAISS index.")
        return "Sorry, I couldn't retrieve the information."
    
    retriever_chain = get_context_retriever_chain(vector_store)
    try:
        # st.write("Debug: Calling Retriever Chain...")
        # Invoke with only the query (language removed)
        response = retriever_chain.invoke({"query": user_input})
        # st.write(f"Debug: Response received - {response}")
        
        result = response.get('result', "Sorry, I couldn't find specific details on that topic.")
        source_urls = [doc.metadata.get("source") for doc in response.get("source_documents", []) if doc.metadata.get("source")]
        
        final_response = f"{result}"
        if source_urls:
            final_response += "\n\nReferences:\n" + "\n".join(f"- [Source]({url})" for url in source_urls)
        return final_response   
    except Exception as e:
        st.error(f"Error occurred: {e}")
        return "Sorry, something went wrong. Please try again later."

# Chat input
user_query = st.chat_input("Type your message here...")

# Audio processing
audio_bytes = audio_recorder("Speak now")
if not audio_bytes:
    st.warning("Please record some audio to proceed.")
else:
    # st.write("Audio recorded. Processing...")
    st.session_state.recorded_audio = audio_bytes
    file_path = bhashini_master.save_audio_as_wav(audio_bytes, directory="output", file_name="last_recording.wav")
    detected_audio_language = fallback_language_code
    # st.write(f"Detected audio language: {detected_audio_language}")
    transcribed_text = bhashini_master.transcribe_audio(audio_bytes, source_language=detected_audio_language)
    if transcribed_text:
        st.write(f"Transcribed Audio: {transcribed_text}")
        response = get_response(transcribed_text)
        st.session_state.chat_history.append(HumanMessage(content=transcribed_text))
        st.session_state.chat_history.append(AIMessage(content=response))
        st.markdown(f"**You:** {transcribed_text}")
        st.markdown(f"🤖 **Mitra:** {response}")
        bhashini_master.speak(response, source_language=detected_audio_language)
        # st.warning("Please record some audio to proceed.")
    else:
        st.write("Error: Audio transcription failed.")

# Process manual text input if available
if user_query:
    # If your Bhashini master has a detect_text_language method, use it; otherwise, use the fallback.
    try:
        detected_text_language = bhashini_master.detect_text_language(user_query)
    except Exception:
        detected_text_language = fallback_language_code
    detected_text_language = fallback_language_code  # Using fallback for consistency
    # st.write(f"Detected text language: {detected_text_language}")   
    with st.spinner("Generating response..."):
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        st.markdown(f"**You:** {user_query}")
        st.markdown(f"🤖 **Mitra:** {response}")
        bhashini_master.speak(response, source_language=language_code)

# Sidebar for Chat History
# st.sidebar.title("Chat History")
footer = """
    <div class="footer">
        <p style="text-align: left;">Copyright © 2024 Citizen Services. All rights reserved.</p>
        <p style="text-align: right;">The responses provided by this chatbot are AI-generated. Please verify with official sources.</p>
    </div>
"""
with st.sidebar:
    st.sidebar.markdown(footer, unsafe_allow_html=True)
    st.title("Chat History")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                st.markdown(f"**You:** {message.content}")
            elif isinstance(message, AIMessage):
                st.markdown(f"🤖 **Mitra:** {message.content}")
    if st.button("Clear Chat History"):
        # Reset chat history
        st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]
        
        # Reset audio-related session state
        if "recorded_audio" in st.session_state:
            del st.session_state["recorded_audio"]
        if "transcribed_text" in st.session_state:
            del st.session_state["transcribed_text"]
        
        # Ensure other relevant session state variables are reset
        # Add any other session state variables that need to be reset
        
        # Delete the saved WAV file if it exists
        audio_file_path = os.path.join(os.getcwd(), "output", "last_recording.wav")
        try:
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
        except Exception as e:
            st.error(f"Failed to delete audio file: {e}")
        
        # Rerun the Streamlit app to ensure complete reset
        st.rerun()

st.markdown(footer, unsafe_allow_html=True)
