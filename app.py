import os
import json
import groq
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from langchain_groq import ChatGroq
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun

# ---------------------------- Load Environment ----------------------------
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") # Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Search engine Chatbot With GROQ"
CHAT_LOG_FILE = "chat_log.json"

# ---------------------------- Utility Functions ----------------------------
def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def initialize_session():
    return [{"role": "assistant", "content": "Hi! I am a helpful assistant. How can I help you?", "time": timestamp()}]

def save_chat_log():
    with open(CHAT_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state["messages"], f, indent=2, ensure_ascii=False)

def load_chat_log():
    if os.path.exists(CHAT_LOG_FILE):
        with open(CHAT_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return initialize_session()
# ---------------------------- UI Setup ----------------------------
st.set_page_config(page_title="LangChain + Groq Chatbot", page_icon="🤖")
st.title("🚀 LangChain + Groq: Chat with Tools & Memory")
st.markdown("Ask me anything! Powered by Groq + LangChain with memory and smart search.")

# ---------------------------- Sidebar ----------------------------
with st.sidebar:
    st.header("Settings")
    default_key = os.getenv("GROQ_API_KEY", "")
    api_key = st.text_input("GROQ API Key", type="password", value=default_key, help="Get Account at: https://console.groq.com/login")

    if "api_key" not in st.session_state or st.session_state.api_key != api_key:
        if not api_key:
            st.session_state.api_valid = False
            st.warning("Please enter your GROQ API Key to use the chatbot.")
            st.stop()
        try:
            client = groq.Groq(api_key=api_key)
            models = client.models.list()
            if models.data:
                st.session_state.api_valid = True
                st.session_state.api_key = api_key
                st.success("✅ GROQ API key is valid!")
            else:
                st.session_state.api_valid = False
                st.error("❌ Invalid API key or empty model list.")
                st.stop()
        except Exception as e:
            st.session_state.api_valid = False
            st.error(f"❌ Validation failed: {str(e).splitlines()[0]}")
            st.markdown(
                '<span style="color:gray; font-size: 0.9em;">🔗 '
                '<a href="https://console.groq.com" target="_blank">GET Free API Key at console.groq.com</a></span>',
                unsafe_allow_html=True
            )
            st.stop()
    else:
        st.success("✅ GROQ API key is already validated.")

    model_name = st.selectbox("Model", ["llama3-8b-8192", "gemma2-9b-it"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.number_input("Max Tokens", 50, 3000, 150)

    if st.button("Clear Chat"):
        st.session_state.messages = initialize_session()
        if os.path.exists(CHAT_LOG_FILE):
            os.remove(CHAT_LOG_FILE)
        st.rerun()



# ---------------------------- Tool Caching ----------------------------
tool_cache = {}

def cached_search(query, tool_instance, **kwargs):
    if query in tool_cache:
        return tool_cache[query]
    result = tool_instance.run(query, **kwargs)
    tool_cache[query] = result
    return result

class CachedDuckDuckGoSearchRun(DuckDuckGoSearchRun):
    def run(self, query: str, **kwargs) -> str:
        return cached_search(query, super(), **kwargs)

class CachedWikipediaQueryRun(WikipediaQueryRun):
    def run(self, query: str, **kwargs) -> str:
        return cached_search(query, super(), **kwargs)

class CachedArxivQueryRun(ArxivQueryRun):
    def run(self, query: str, **kwargs) -> str:
        return cached_search(query, super(), **kwargs)

# ---------------------------- Tool Setup ----------------------------
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
arxiv = CachedArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
wiki = CachedWikipediaQueryRun(api_wrapper=wiki_wrapper)

search_engine = CachedDuckDuckGoSearchRun(name="Search", description="Search the web for information")

tools = [search_engine, arxiv, wiki]

# ---------------------------- Embedding Model ----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def find_similar_response(new_prompt, history, threshold=0.5):
    user_prompts = [m for m in history if m["role"] == "user"]
    if not user_prompts:
        return None
    new_embedding = embed_model.encode(new_prompt, convert_to_tensor=True)
    existing_embeddings = embed_model.encode([m["content"] for m in user_prompts], convert_to_tensor=True)
    similarities = util.cos_sim(new_embedding, existing_embeddings)[0]
    best_idx = int(similarities.argmax())
    best_score = float(similarities[best_idx])
    if best_score >= threshold:
        for i in range(len(history)):
            if history[i]["role"] == "user" and history[i]["content"] == user_prompts[best_idx]["content"]:
                if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
                    return history[i + 1]["content"]
    return None
# ---------------------------- Session State ----------------------------
if "initialized" not in st.session_state:
    if os.path.exists(CHAT_LOG_FILE):
        os.remove(CHAT_LOG_FILE)
    st.session_state.initialized = True

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_log()

# ---------------------------- Display Chat History ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f"*{msg['time']}*\n\n{msg['content']}")

# ---------------------------- Prompt Handler ----------------------------
def handle_prompt(prompt):
    new_user_msg = {"role": "user", "content": prompt, "time": timestamp()}
    st.session_state.messages.append(new_user_msg)
    save_chat_log()
    st.chat_message("user").markdown(f"*{new_user_msg['time']}*\n\n{prompt}")

    llm = ChatGroq(
        groq_api_key=st.session_state.api_key,
        model_name=model_name,
        streaming=True,
        max_tokens=max_tokens,
        temperature=temperature
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            memory.chat_memory.add_user_message(msg["content"])
        else:
            memory.chat_memory.add_ai_message(msg["content"])

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        handle_parsing_errors=True,
        verbose=False
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = find_similar_response(prompt, st.session_state.messages, threshold=0.5)
        if response:
            response = f"( Reused previous answer)\n\n{response}"
        else:
            try:
                response = agent.run(prompt, callbacks=[st_cb])
            except Exception as e:
                response = f" Error: {str(e)}"
        new_ai_msg = {"role": "assistant", "content": response, "time": timestamp()}
        st.session_state.messages.append(new_ai_msg)
        st.markdown(f"*{new_ai_msg['time']}*\n\n{response}")
        save_chat_log()

# ---------------------------- Suggested Prompt Handler ----------------------------
if "suggested_input" in st.session_state:
    prompt = st.session_state.pop("suggested_input")
    handle_prompt(prompt)

# ---------------------------- Chat Input ----------------------------
if prompt := st.chat_input("Ask me anything!"):
    handle_prompt(prompt)

# ---------------------------- Chat Log Download ----------------------------
if os.path.exists(CHAT_LOG_FILE):
    try:
        with open(CHAT_LOG_FILE, "r", encoding="utf-8") as f:
            chat_json = f.read()
        st.sidebar.download_button("⬇️ Download Updated Chat Log", chat_json, "chat_log.json", "application/json")
    except Exception as e:
        st.error(f"❌ Failed to read chat log: {e}")
else:
    st.sidebar.info("Chat log will appear here after your first question.")

# ---------------------------- Chat History Viewer ----------------------------
with st.expander("🧠 Chat History (for reference)", expanded=False):
    for msg in st.session_state["messages"]:
        role_label = "👤 You" if msg["role"] == "user" else "🤖 Assistant"
        st.markdown(f"**{role_label}**  \n*{msg['time']}*\n\n{msg['content']}")

# ---------------------------- Prompt Suggestions ----------------------------
st.markdown("---")
st.markdown("### 💡 Try these:")
suggestions = [
    "What is LangChain?",
    "Explain Groq's LPU tech",
    "How to learn AI?",
    "Write a poem about robots"
]

cols = st.columns(2)
for i, suggestion in enumerate(suggestions):
    with cols[i % 2]:
        if st.button(suggestion):
            st.session_state["suggested_input"] = suggestion
            st.rerun()

st.markdown("---")
st.markdown("Made with ❤️ using LangChain, Groq & Streamlit")

