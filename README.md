# 🤖 LangChain + Groq Chatbot with Tools, Memory, and Smart Search

A powerful, interactive chatbot built using [LangChain](https://www.langchain.com/), [Groq](https://groq.com/), and [Streamlit](https://streamlit.io/). This chatbot supports tool usage (ArXiv, Wikipedia, DuckDuckGo), conversation memory, semantic response reuse, and persistent chat logs — all in a clean UI.

---

## 🚀 Features

- **Groq LLM Support**: Choose from fast, low-latency models like `llama3-8b-8192`, `gemma2-9b-it`
- **Streaming Chat**: See responses stream live from Groq.
- **Tool Integration**: 
  - 🔍 DuckDuckGo for web search  
  - 📚 Wikipedia API  
  - 📄 ArXiv API for scientific papers
- **Semantic Memory**: Avoid repeated answers with vector similarity matching.
- **Session History**: Keeps chat memory across prompts using LangChain’s `ConversationBufferMemory`.
- **Persistent Logs**: Saves chat to `chat_log.json` after every message.
- **Clear Chat**: Fully reset the session and delete logs.
- **Chat Log Download**: Easily download your chat as a JSON file.

---

## 🧠 How It Works

1. **API Key Input & Validation**:
   - You must provide a valid GROQ API key in the sidebar.
   - If the key is invalid or empty:
     - You'll see a warning or error message.
     - The app halts further interaction until fixed.
   - Once validated, the key is stored in session to avoid revalidation on rerun.

2. **Conversation Management**:
   - Memory is handled with LangChain's `ConversationBufferMemory`.
   - It preserves back-and-forth flow during multi-turn conversations.

3. **Semantic Reuse**:
   - Your new prompt is compared with all previous prompts using SentenceTransformers.
   - If similarity > 0.5, the assistant reuses its earlier response to save compute.

4. **Chat Logging**:
   - All messages (user + assistant) are saved to `chat_log.json`.
   - Chat history is replayed on page refresh.
   - You can download the latest chat history via a sidebar button.

5. **Clear Chat**:
   - Pressing **Clear Chat**:
     - Wipes chat from memory.
     - Deletes `chat_log.json`.
     - Resets the app to initial greeting state.

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/ad-archishman12/2-Search-Engine-Groq-Chatbot.git
cd 2-Search-Engine-Groq-Chatbot
```

### 2. Install dependencies
Ensure Python 3.8+ is installed. Then run:

bash
Copy
Edit
pip install -r requirements.txt
If requirements.txt is missing, you can install manually:

bash
Copy
Edit
pip install streamlit python-dotenv groq sentence-transformers langchain langchain-community

### 3. Create .env file
env
Copy
Edit
GROQ_API_KEY=your_groq_key_here
LANGCHAIN_API_KEY=your_langsmith_key_here  # Optional for Langsmith tracking

### 4. Run the app
streamlit run app.py

📝 Suggested Prompts
Use the sidebar or suggested buttons like:

What is LangChain?

Explain Groq's LPU tech

How to learn AI?

Write a poem about robots

📂 File Structure
├── app.py               # Main Streamlit app
├── chat_log.json        # Auto-saved chat history
├── .env                 # API keys (not committed)
├── README.md            # You're reading it!

🛑 Known Behaviors
Empty or Invalid API Key:

The app halts and shows an appropriate message.

A retry is allowed upon correcting the input.

## Clear Chat:
Deletes the chat_log.json.

Fully resets session state and history.

## 💡 Future Ideas
Add OpenAI or Anthropic model support

User authentication with chat ownership

Enable prompt chaining or document Q&A

Add images, charts, or multimedia output
