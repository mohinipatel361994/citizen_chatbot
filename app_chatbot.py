import os, re, logging, uuid, base64, json,ast,tiktoken
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from rapidfuzz import process, fuzz
from bhashini_services1 import Bhashini_master
from audio_recorder_streamlit import audio_recorder
from PIL import Image
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from prompts import scheme_prompt, prompt_template, refine_gemini
import google.generativeai as genai
from vertexai.language_models import TextGenerationModel
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-xlm-r-multilingual-v1"
)
# Load schemes data from a JSON file
with open("myscheme_json/all_schemes_madhya_pradesh.json", "r", encoding="utf-8") as f:
    data = json.load(f)

schemes = data.get("Schemes", [])
model = genai.GenerativeModel("models/gemini-2.0-flash-lite")

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler()
    ],
    force=True
)

logging.info("Application started.")
# Helper class to force UTF-8 encoding

def normalize_text(text):
    """Lowercase and remove extra spaces. Handles None input safely."""
    if not text:
        logging.warning("normalize_text received None or empty input.")
        return ""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    logging.debug(f"Normalized text: {normalized}")
    return normalized

def gemini_translate_one_word(text, source_language, target_language):
    prompt = (
        f"Translate the following {source_language} word to {target_language}. "
        f"Return only the translated sentenc, nothing else:\n\n{text}"
    )

    response = model.generate_content(prompt)
    return response.text.strip()

def correct_spelling(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # remove extra whitespace
    text = text.strip()
    text = text.replace("sikho", "seekho")  # simple manual correction
    text = text.replace("Kamau", "Kamao")  # correct spelling
    text = text.replace("yojna", "yojana")  # correct spelling
    text = text.replace("Yojana", "s")  # correct spelling
    return text
# Initialize session state for chat history and session ID
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="नमस्ते, मैं एक बॉट हूँ। मैं आपकी कैसे मदद कर सकता हूँ? ")]
    logging.info("Initialized chat history.")
    
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
    """
    <div style="
        background: linear-gradient(90deg, #ffecd2 0%, #fcb69f 100%);
        border: 2px solid #f79489;
        border-radius: 10px;
        padding: 20px;
        font-size: 18px;
        font-weight: 500;
        line-height: 1.6;
        color: #4a2c2a;
        box-shadow: 2px 2px 10px rgba(252, 182, 159, 0.5);
        ">
        <b>जन सेवा सहायक</b> एक AI-आधारित चैटबॉट है जो नागरिकों को <b>मध्य प्रदेश सरकार की योजनाओं और सेवाओं</b> के बारे में जानकारी देने के लिए बनाया गया है। 
        यह आपको विभिन्न सरकारी योजनाओं, लाभ, प्रक्रियाओं और सेवाओं की विस्तृत जानकारी प्रदान करता है।  
        यदि आपको किसी योजना के लिए पात्रता, आवेदन प्रक्रिया, दस्तावेज़ों की आवश्यकताएँ या अन्य कोई सहायता चाहिए, तो यह चैटबॉट आपकी सहायता के लिए तैयार है।  
        <br><br>
        <b>आप कैसे मदद ले सकते हैं?</b>  
        <br>1. किसी भी सरकारी योजना की जानकारी प्राप्त करें  
        <br>2. आवेदन प्रक्रिया और पात्रता शर्तें जानें  
        <br>3. दस्तावेज़ और आवश्यक कागजात की जानकारी लें  
    </div>
    """,
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
logging.info(f"Selected language: {selected_language}")

google_api_key = st.secrets["secret_section"]["google_api_key"]
api_key = st.secrets["secret_section"]["openai_api_key"]
bhashini_url = st.secrets["secret_section"]["bhashini_url"]
bhashini_authorization_key = st.secrets["secret_section"]["bhashini_authorization_key"]
bhashini_ulca_api_key = st.secrets["secret_section"]["bhashini_ulca_api_key"]
bhashini_ulca_userid = st.secrets["secret_section"]["bhashini_ulca_userid"]

# google_api_key = os.getenv("google_api_key")
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
PERSIST_DIR = os.path.join(os.getcwd(), "faiss_index_pagging")
if not os.path.exists(PERSIST_DIR):
    logging.error("❌ FAISS index not found! Rebuild it first.")
    print("❌ FAISS index not found! Rebuild it first.")

def load_faiss_vectorstore():
    """Load FAISS vector store from disk and verify dimensions."""
    if not os.path.exists(PERSIST_DIR):
        st.error("FAISS index not found. Please rebuild the FAISS index using the correct embedding model.")
        logging.error("FAISS index not found in expected directory.")
        return None

    expected_dim = len(embedding_model.embed_query("test query"))
    
    try:
        vector_store = FAISS.load_local(PERSIST_DIR, embedding_model, allow_dangerous_deserialization=True)
        if vector_store.index.d != expected_dim:
            st.error(f"Dimension mismatch: expected {expected_dim}, but index has {vector_store.index.d}. Please rebuild the FAISS index.")
            logging.error(f"Dimension mismatch: expected {expected_dim}, but got {vector_store.index.d}.")
            return None
        logging.info("FAISS vector store loaded successfully.")
        return vector_store
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        logging.error(f"Failed to load FAISS index: {e}")
        return None

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
        
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(model="gpt-4", api_key=api_key, temperature=0.3)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
                ),
        }
    )
    logging.info("Context retriever chain created.")
    return qa_chain

genai.configure(api_key=google_api_key)
# def get_fuzzy_matches(query, scheme_names, threshold=85, top_n=3):
#     matches = process.extract(query, scheme_names, scorer=fuzz.token_sort_ratio, limit=top_n)
#     return [match[0] for match in matches if match[1] >= threshold]
def regex_search_schemes(query, schemes):
    try:
        scheme_names = [scheme.get("Scheme Name", "") for scheme in schemes if isinstance(scheme, dict)]
        # filtered_scheme_names = get_fuzzy_matches(query, scheme_names)
        # print("filtered_scheme_names",filtered_scheme_names)
        # Step 1: Try to match a scheme name directly
        name_match_prompt = (
            f"You are given a list of government schemes:\n{scheme_names}\n\n"
            f"The user asked about this scheme: '{query}'\n\n"
            f"Your task is to identify the best matching scheme name **only if the name is explicitly mentioned or almost identical (typos allowed)**.\n\n"
            f"⚠️ STRICT RULES:\n"
            f"- Do NOT match based on topic, meaning, or context.\n"
            f"- DO NOT assume the user meant a different scheme with a similar name.\n"
            f"- For example, treat 'Ladli Behna Yojana' and 'Ladli Laxmi Yojana' as **completely different**.\n"
            f"- Only return a scheme from the list that is spelled the same (or almost the same with minor typo).\n\n"
            f"❌ If the user's input does NOT match any scheme name from the list, return: None\n"
            f"✅ Return only ONE scheme name from the list above, exactly as it appears, or 'None'."
        )
        response = model.generate_content(name_match_prompt)
        matched_name = response.text.strip() if hasattr(response, 'text') else None

        if matched_name and matched_name.lower() != "none":
            for scheme in schemes:
                if not isinstance(scheme, dict):
                    continue
                if scheme.get("Scheme Name", "").strip().lower() == matched_name.lower():
                    print("matched scheme name ",scheme)
                    return [scheme]  # Return as list

        # Step 2: Match based on eligibility criteria
        eligibility_data = [
            {
                "Scheme Name": scheme.get("Scheme Name", ""),
                "Eligibility": scheme.get("Eligibility", "")
            }
            for scheme in schemes if isinstance(scheme, dict)
        ]

        eligibility_prompt = (
            f"You are given a list of government schemes and their eligibility criteria:\n\n"
            f"{eligibility_data}\n\n"
            f"User query: '{query}'\n\n"
            f"Return only a valid Python list of exact scheme names that match the eligibility criteria in the query. "
            f"Only return a list like ['Scheme A', 'Scheme B',...], nothing else. If nothing matches, return []."
        )

        eligibility_response = model.generate_content(eligibility_prompt)
        if hasattr(eligibility_response, "text"):
            response_text = eligibility_response.text.strip()
            print("Gemini raw response:\n", response_text) 
        return response_text
   
    except Exception as e:
        logging.exception("Gemini scheme matching failed:")
        return []

def load_scheme_vectorstore(scheme_name):
    try:
        # Assume the FAISS indexes are saved under faiss_index_paging/{scheme_name}/
        scheme_folder = f"faiss_index_paging/{scheme_name.lower().replace(' ', '_')}"
        
        if not os.path.exists(scheme_folder):
            logging.error(f"Vector store folder does not exist for scheme: {scheme_folder}")
            return None

        # Load FAISS vectorstore using HuggingFace embeddings
        vector_store = FAISS.load_local(folder_path=scheme_folder, embeddings=embedding_model)
        
        logging.info(f"Successfully loaded vector store for scheme: {scheme_name}")
        return vector_store
    except Exception as e:
        logging.error(f"Error loading vector store for scheme {scheme_name}: {e}")
        return None
    
vector_store = load_faiss_vectorstore()     
MAX_MODEL_TOKENS = 8192
BUFFER_FOR_PROMPT_AND_RESPONSE = 1500
MAX_CONTEXT_TOKENS = MAX_MODEL_TOKENS - BUFFER_FOR_PROMPT_AND_RESPONSE

# Token estimation function
def num_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Split large JSON string into token-safe chunks
def split_json_chunks(json_data, max_tokens, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for scheme in json_data:
        scheme_text = json.dumps(scheme, indent=2, ensure_ascii=False)
        token_count = len(encoding.encode(scheme_text))
        if current_tokens + token_count <= max_tokens:
            current_chunk.append(scheme)
            current_tokens += token_count
        else:
            chunks.append(current_chunk)
            current_chunk = [scheme]
            current_tokens = token_count
    if current_chunk:
        chunks.append(current_chunk)
    return chunks
def truncate_to_token_limit(text, max_tokens=7500, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)  
def refine_with_gemini(user_query, raw_response):
    try:
        prompt=refine_gemini.format(
                    raw_response=raw_response,
                    user_query=user_query
                )
        result = model.generate_content(prompt)
        return result.text.strip()

    except Exception as e:
        logging.error(f"Gemini refinement failed: {e}")
        return raw_response  # fallback to original
def get_response(user_input):
    norm_query = normalize_text(user_input)
    corrected_query = correct_spelling(norm_query)

    matched_schemes = regex_search_schemes(corrected_query, schemes)
    print("matched_schemes", matched_schemes)

    # Handle fallback: if matched_schemes is a list of strings, convert to dicts
    if matched_schemes and all(isinstance(s, str) for s in matched_schemes):
        matched_schemes = [
            scheme for scheme in schemes
            if scheme.get("Scheme Name", "").strip() in matched_schemes
        ]

    if matched_schemes and all(isinstance(s, dict) for s in matched_schemes):
        try:
            json_chunks = split_json_chunks(matched_schemes, max_tokens=MAX_CONTEXT_TOKENS)
            # llm = ChatOpenAI(model="gpt-4", api_key=api_key, temperature=0.3)

            chunk_responses = []
            for i, chunk in enumerate(json_chunks):
                chunk_json = json.dumps(chunk, indent=2, ensure_ascii=False)
                prompt = scheme_prompt.format(
                    regex_result=chunk_json,
                    corrected_query=corrected_query
                )

                if num_tokens(prompt) > MAX_MODEL_TOKENS:
                    logging.warning(f"Skipping chunk {i+1} due to token limit.")
                    continue

                response = model.generate_content(prompt)
                print(f"chunk_responses{i}", response.text)
                chunk_responses.append(response.text.strip())

            final_response = "\n\n".join(chunk_responses) if chunk_responses else "कोई योजना जानकारी नहीं मिली।"

            try:
                final_response = refine_with_gemini(corrected_query, final_response)
            except Exception as e:
                logging.error(f"Refinement with Gemini failed: {e}")

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
        retriever_chain = get_context_retriever_chain(vector_store)

        try:
            response = retriever_chain.invoke({
                "query": corrected_query,
                "chat_history": chat_history_str,
            })

            result = response.get('result', "Sorry, I couldn't find specific details on that topic.")
            source_urls = [
                doc.metadata.get("source")
                for doc in response.get("source_documents", [])
                if doc.metadata.get("source")
            ]

            final_response = result
            if source_urls:
                final_response += "\n\nReferences:\n" + "\n".join(f"- [Source]({url})" for url in source_urls)

            logging.info("Response generated successfully.")
            log_chat_history()
            return final_response

        except Exception as e:
            st.error(f"Error occurred: {e}")
            logging.error(f"Error in get_response: {e}")
            return "Sorry, something went wrong. Please try again later."

if "audio_processed" not in st.session_state:
    st.session_state.audio_processed = False

# Audio processing
col1, col2 = st.columns([0.8, 0.2])

with col1:
    user_query = st.chat_input("अपने संदेश टाइप करें...")  # Text input

with col2:
    audio_bytes = audio_recorder("रिकॉर्ड करें")  # Microphone button

if not audio_bytes:
    st.warning("अपना प्रश्न टाइप करके या रिकॉर्ड करके पूछें।")
    logging.info("No audio recorded.")
else:
    st.session_state.recorded_audio = audio_bytes
    file_path = bhashini_master.save_audio_as_wav(audio_bytes, directory="output", file_name="last_recording.wav")
    logging.info(f"Audio saved at {file_path}")

    transcribed_text = bhashini_master.transcribe_audio(audio_bytes, source_language=language_code)

    if transcribed_text:
        # Translate to English
        translated_input = gemini_translate_one_word(
            transcribed_text,
            source_language=language_code,
            target_language="en"
        )
        
        with st.spinner("Generating response..."):
            response_in_english = get_response(translated_input)

            # Translate back to original language
            translated_response = gemini_translate_one_word(
                response_in_english,
                source_language="en",
                target_language=language_code
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

            bhashini_master.speak(translated_response, source_language=language_code)
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

    # Translate to English
    translated_input = gemini_translate_one_word(
        user_query,
        source_language=language_code,
        target_language="English"
    )
    print("user_query",user_query)
    print("translated_input",translated_input)

    with st.spinner("Generating response..."):
        response_in_english = get_response(translated_input)

        # Translate back to original language
        translated_response = gemini_translate_one_word(
            response_in_english,
            source_language="English",
            target_language=language_code
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

        bhashini_master.speak(translated_response, source_language=language_code)
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
