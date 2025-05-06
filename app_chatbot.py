import os,json
import re
import logging
import uuid
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
from rapidfuzz import process, fuzz
import google.generativeai as genai
from prompts import scheme_prompt, prompt_template
with open("myscheme_json/all_schemes_madhya_pradesh.json", "r", encoding="utf-8") as f:
    data = json.load(f)

schemes = data.get("Schemes", [])
# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logging.info("Application started.")

st.set_page_config(page_title="जन सेवा सहायक", page_icon="image/Emblem_of_Madhya_Pradesh.svg", layout="wide")
common_variants = {
    "seekho": "sikho",
    "Kamao": "Kamau",    
    "Yojana": "yojna",
    "yojna": "scheme",
}

def normalize_text(text):
    """Lowercase and remove extra spaces."""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    logging.debug(f"Normalized text: {normalized}")
    return normalized

def correct_spelling(text, variant_dict, threshold=80):
    """
    For each word in text, check for close matches in variant_dict.
    If the fuzzy match score exceeds the threshold, replace the word.
    """
    words = text.split()
    corrected_words = []
    for word in words:
        match, score, _ = process.extractOne(word, variant_dict.keys(), scorer=fuzz.ratio)
        if score >= threshold:
            corrected_words.append(variant_dict[match])
            logging.debug(f"Corrected '{word}' to '{variant_dict[match]}' (score: {score}).")
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)
    
# Add background image from a local file
def add_bg_from_local(image_file, opacity=0):
    try:
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
        logging.info("Background image added successfully.")
    except Exception as e:
        logging.error(f"Failed to add background image: {e}")

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
                <h1 style="color:#000080; margin-bottom: 0;">🤖 जन सेवा सहायक</h1>
                <p style="font-size: 18px; font-weight: 600; margin-top: 5px;">सेवा सहायक: मध्य प्रदेश सरकार की योजनाओं और सेवाओं के लिए आपका डिजिटल सहायक</p>
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
    """<div class="title" style="font-size: 18px; font-weight: 500; line-height: 1.6;">
    <b>जन सेवा सहायक</b> एक AI-आधारित चैटबॉट है जो नागरिकों को <b>मध्य प्रदेश सरकार की योजनाओं और सेवाओं</b> के बारे में जानकारी देने के लिए बनाया गया है। 
    यह आपको विभिन्न सरकारी योजनाओं, लाभ, प्रक्रियाओं और सेवाओं की विस्तृत जानकारी प्रदान करता है।  
    यदि आपको किसी योजना के लिए पात्रता, आवेदन प्रक्रिया, दस्तावेज़ों की आवश्यकताएँ या अन्य कोई सहायता चाहिए, तो यह चैटबॉट आपकी सहायता के लिए तैयार है।  
    <br>
    <b>आप कैसे मदद ले सकते हैं?</b>  
    <br>1. किसी भी सरकारी योजना की जानकारी प्राप्त करें  
    <br>2. आवेदन प्रक्रिया और पात्रता शर्तें जानें  
    <br>3. दस्तावेज़ और आवश्यक कागजात की जानकारी लें  
    </div>""",
    unsafe_allow_html=True
)

# Initialize session state for chat history and session ID
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="नमस्ते, मैं एक बॉट हूँ। मैं आपकी कैसे मदद कर सकता हूँ? ")]
    logging.info("Initialized chat history.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logging.info(f"New session started with session ID: {st.session_state.session_id}")

# Load environment variables
load_dotenv()
logging.info("Environment variables loaded.")

# Available languages for selection
languages = {
    "हिन्दी": "hi",
    "English": "en",
}
selected_language = st.selectbox("प्रश्न की भाषा:", options=list(languages.keys()))
language_code = languages[selected_language]
fallback_language_code = languages[selected_language]
logging.info(f"Selected language: {selected_language}")

google_api_key = st.secrets["secret_section"]["google_api_key"]
api_key = st.secrets["secret_section"]["openai_api_key"]
bhashini_url = st.secrets["secret_section"]["bhashini_url"]
bhashini_authorization_key = st.secrets["secret_section"]["bhashini_authorization_key"]
bhashini_ulca_api_key = st.secrets["secret_section"]["bhashini_ulca_api_key"]
bhashini_ulca_userid = st.secrets["secret_section"]["bhashini_ulca_userid"]

# api_key = os.getenv("openai_api_key")
# bhashini_url = os.getenv("bhashini_url")
# bhashini_authorization_key = os.getenv("bhashini_authorization_key")
# bhashini_ulca_api_key = os.getenv("bhashini_ulca_api_key")
# bhashini_ulca_userid = os.getenv("bhashini_ulca_userid")
# Initialize Bhashini master for transcription
bhashini_master = Bhashini_master(
    url=bhashini_url,
    authorization_key=bhashini_authorization_key,
    ulca_api_key=bhashini_ulca_api_key,
    ulca_userid=bhashini_ulca_userid
)
logging.info("Bhashini master initialized.")

# Directory for FAISS index
PERSIST_DIR = os.path.join(os.getcwd(), "faiss_index_eoffice")
if not os.path.exists(PERSIST_DIR):
    logging.error("❌ FAISS index not found! Rebuild it first.")
    print("❌ FAISS index not found! Rebuild it first.")

def load_faiss_vectorstore():
    """Load FAISS vector store from disk and verify dimensions."""
    if not os.path.exists(PERSIST_DIR):
        st.error("FAISS index not found. Please rebuild the FAISS index using the correct embedding model.")
        logging.error("FAISS index not found in expected directory.")
        return None
    # model_name = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
    # embeddings = HuggingFaceEmbeddings(model_name=model_name)
    # expected_dim = len(embeddings.embed_query("test query"))
    
    # try:
    #     vector_store = FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)
    #     if vector_store.index.d != expected_dim:
    #         st.error(f"Dimension mismatch: expected {expected_dim}, but index has {vector_store.index.d}. Please rebuild the FAISS index.")
    #         logging.error(f"Dimension mismatch: expected {expected_dim}, but got {vector_store.index.d}.")
    #         return None
    #     logging.info("FAISS vector store loaded successfully.")
    #     return vector_store
    # except Exception as e:
    #     st.error(f"Failed to load FAISS index: {e}")
    #     logging.error(f"Failed to load FAISS index: {e}")
    #     return None

def log_chat_history():
    # Log the session ID and a simplified version of the chat history.
    chat_log = [
        {"role": msg.__class__.__name__, "content": msg.content}
        for msg in st.session_state.chat_history
    ]
    logging.info(f"Session {st.session_state.session_id} chat history: {chat_log}")
def get_chat_history_string(max_turns=5):
    """
    Returns the last max_turns of chat history as a single string.
    This helps the model recall previous interactions.
    """
    history_lines = []
    # Only take the last `max_turns` pairs (or messages) for brevity.
    # Adjust slicing as necessary.
    for msg in st.session_state.chat_history[-max_turns:]:
        role = "User" if isinstance(msg, HumanMessage) else "Bot"
        history_lines.append(f"{role}: {msg.content}")
    return "\n".join(history_lines)
    
def get_context_retriever_chain(vector_store, language_code):
    llm = ChatOpenAI(model="gpt-4", api_key=api_key, temperature=0.3)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    test_query = "What services are available for citizens?"
    retrieved_docs = retriever.get_relevant_documents(test_query)
    logging.info(f"Retrieved {len(retrieved_docs)} documents for test query.")    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt_template,
        }
    )
    logging.info("Context retriever chain created.")
    return qa_chain
vector_store = load_faiss_vectorstore()
genai.configure(api_key=google_api_key)
def regex_search_schemes(query, schemes):
    """
    Uses Gemini to find the best matching scheme name from a list of schemes.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # Or the version you verified

        scheme_names = [scheme.get("Scheme Name", "") for scheme in schemes if isinstance(scheme, dict)]

        # Construct prompt
        prompt = (
            f"Given the following list of scheme names:\n{scheme_names}\n\n"
            f"Which one best matches the user query: '{query}'?\n"
            f"Respond with only the matching scheme name exactly as listed above."
        )

        logging.info(f"Sending prompt to Gemini: {prompt}")
        response = model.generate_content(prompt)
        matched_name = response.text.strip() if hasattr(response, 'text') else None

        # Find the corresponding full scheme dict
        for scheme in schemes:
            if scheme.get("Scheme Name", "").strip().lower() == matched_name.lower():
                return scheme

        logging.warning(f"Gemini matched scheme name not found in original list: {matched_name}")
        return None

    except Exception as e:
        logging.exception("Gemini scheme matching failed:")
        return None
def get_response(user_input):
    norm_query = normalize_text(user_input)
    corrected_query = correct_spelling(norm_query, common_variants)
    regex_result = regex_search_schemes(corrected_query, schemes)
    regex_result=json.dumps(regex_result, indent=2, ensure_ascii=False)
    if regex_result:
        scheme_name = regex_result.get("Scheme Name", "Unnamed Scheme")
        try:
            llm = ChatOpenAI(model="gpt-4", api_key=api_key, temperature=0.3)
            result = llm.invoke({"input": scheme_prompt})
            final_response = result.content
        except Exception as e:
            logging.error(f"LLM invocation failed: {e}")
            final_response = "योजना विवरण निकालने में त्रुटि हुई। कृपया बाद में प्रयास करें।"
        
        log_chat_history()
        return final_response
    else:
        logging.warning("No scheme identified via regex.")
        if not vector_store:
            st.error("Vector store not found. Please rebuild the FAISS index.")
            logging.error("Vector store not found.")
            return "Sorry, I couldn't retrieve the information."
        chat_history_str = get_chat_history_string(max_turns=5)
        retriever_chain = get_context_retriever_chain(vector_store, language_code)
        try:
            response = retriever_chain.invoke({"query": corrected_query, "chat_history": chat_history_str,})
            result = response.get('result', "Sorry, I couldn't find specific details on that topic.")
            source_urls = [doc.metadata.get("source") for doc in response.get("source_documents", []) if doc.metadata.get("source")]
            final_response = f"{result}"
            if source_urls:
                final_response += "\n\nReferences:\n" + "\n".join(f"- [Source]({url})" for url in source_urls)
            logging.info("Response generated successfully.")
            # Log the updated chat history after response generation.
            log_chat_history()
            return final_response
        except Exception as e:
            st.error(f"Error occurred: {e}")
            logging.error(f"Error in get_response: {e}")
            return "Sorry, something went wrong. Please try again later."

if "audio_processed" not in st.session_state:
    st.session_state.audio_processed = False

# Chat input
# user_query = st.chat_input("Type your message here...")

# Audio processing
col1, col2 = st.columns([0.8, 0.2])

with col1:
    user_query = st.chat_input("अपने संदेश टाइप करें...")  # Text input

with col2:
    audio_bytes = audio_recorder("रिकॉर्ड करें")  # Microphone button

if not audio_bytes:
    st.warning("कृपया आगे बढ़ने के लिए ऑडियो रिकॉर्ड करें।")
    logging.info("No audio recorded.")
else:
    st.session_state.recorded_audio = audio_bytes
    file_path = bhashini_master.save_audio_as_wav(audio_bytes, directory="output", file_name="last_recording.wav")
    logging.info(f"Audio saved at {file_path}")

    detected_audio_language = fallback_language_code
    transcribed_text = bhashini_master.transcribe_audio(audio_bytes, source_language=detected_audio_language)

    if transcribed_text:
        # Translate to English
        translated_input = bhashini_master.translate_text(
            transcribed_text,
            source_language=detected_audio_language,
            target_language="en"
        )
        
        with st.spinner("Generating response..."):
            response_in_english = get_response(translated_input)

            # Translate back to original language
            translated_response = bhashini_master.translate_text(
                response_in_english,
                source_language="en",
                target_language=detected_audio_language
            )

            # Show chat in UI
            st.session_state.chat_history.append(HumanMessage(content=transcribed_text))
            if translated_response is not None:
                st.session_state.chat_history.append(AIMessage(content=translated_response))
            else:
                st.warning("Translation failed or returned no result.")

            st.markdown(f"**You:** {transcribed_text}")
            st.markdown(f"**Translated:** {translated_input}")
            st.markdown(f"🤖 **सेवा सहायक:** {translated_response}")

            bhashini_master.speak(translated_response, source_language=detected_audio_language)
            st.session_state.audio_processed = True
            logging.info("Audio processed and response generated.")
    else:
        st.write("Error: Audio transcription failed.")
        logging.error("Audio transcription failed.")

    if "recorded_audio" in st.session_state:
        del st.session_state["recorded_audio"]
        logging.info("Cleared recorded audio from session state.")
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logging.info("Temporary audio file deleted.")
        except Exception as e:
            st.error(f"Failed to delete audio file: {e}")
            logging.error(f"Failed to delete audio file: {e}")

# Manual text input handling
if user_query and not st.session_state.audio_processed:

    detected_text_language = fallback_language_code
    # Translate to English
    translated_input = bhashini_master.translate_text(
        user_query,
        source_language=detected_text_language,
        target_language="en"
    )
    print("user_query",user_query)
    print("translated_input",translated_input)

    with st.spinner("Generating response..."):
        response_in_english = get_response(translated_input)

        # Translate back to original language
        translated_response = bhashini_master.translate_text(
            response_in_english,
            source_language="en",
            target_language=detected_text_language
        )

        # Show chat in UI
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        if translated_response is not None:
            st.session_state.chat_history.append(AIMessage(content=translated_response))
        else:
            st.warning("Translation failed or returned no result.")
            
        st.markdown(f"**You:** {user_query}")
        st.markdown(f"**Translated:** {translated_input}")
        st.markdown(f"🤖 **सेवा सहायक:** {translated_response}")

        bhashini_master.speak(translated_response, source_language=detected_text_language)
        logging.info("Processed manual text input.")

# Sidebar for Chat History
footer = """
    <div class="footer">
        <p style="text-align: left;">Copyright © 2025 Citizen Services. All rights reserved.</br>The responses provided by this chatbot are AI-generated. Please verify with official sources.</p>
    </div>
"""
if 'refresh' not in st.session_state:
    st.session_state.refresh = 0

def refresh_state():
    st.session_state.refresh += 1
    logging.info(f"Refresh state updated: {st.session_state.refresh}")

with st.sidebar:
    st.sidebar.markdown(footer, unsafe_allow_html=True)
    st.title("Chat History")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                st.markdown(f"**You:** {message.content}")
            elif isinstance(message, AIMessage):
                st.markdown(f"🤖 **सेवा सहायक:** {message.content}")
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = [AIMessage(content="नमस्ते, मैं एक बॉट हूँ। मैं आपकी कैसे मदद कर सकता हूँ? ")]
        logging.info(f"Session {st.session_state.session_id}: Chat history cleared.")
        if "recorded_audio" in st.session_state:
            del st.session_state["recorded_audio"]
        if "transcribed_text" in st.session_state:
            del st.session_state["transcribed_text"]
        audio_file_path = os.path.join(os.getcwd(), "output", "last_recording.wav")
        try:
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
                logging.info("Cleared temporary audio file on chat history clear.")
        except Exception as e:
            st.error(f"Failed to delete audio file: {e}")
            logging.error(f"Failed to delete audio file on chat history clear: {e}")
        # Log the cleared chat history
        log_chat_history()

st.markdown(footer, unsafe_allow_html=True)
logging.info("Application finished rendering.")
