# -----------------------------------------------------
# This version includes:
# 1. Instant hybrid greetings
# 2. Ultra-fast retrieval with SentenceTransformer
# 3. Deque-based chat history for context-aware responses
# 4. Clean answers without file references
# -----------------------------------------------------

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
import logging
import time
import re
from typing import List
from collections import deque
from sentence_transformers import SentenceTransformer

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "csv_data"
LLM_MODEL = "mistral"

# FastAPI app
app = FastAPI(title="FAST CSV Chatbot", version="3.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global objects
_llm = None
_vectorstore = None
_embedder = None
_initialized = False
_chat_history = deque(maxlen=5)  # Store last 5 user-bot exchanges


# ---------- FAST EMBEDDER CLASS ----------
class FastEmbedder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


# ---------- SINGLETON INITIALIZER ----------
def init_singleton():
    global _llm, _vectorstore, _embedder, _initialized

    if _initialized:
        return True

    print("\n=== Initializing Fast Chatbot ===")

    # Load fast LLM (only once)
    try:
        _llm = Ollama(model=LLM_MODEL, temperature=0.1)
        print("LLM loaded successfully.")
    except Exception as e:
        print(f"LLM load failed: {e}")
        return False

    # Load embedder
    try:
        _embedder = FastEmbedder()
        print("Embedder loaded.")
    except Exception as e:
        print(f"Embedding error: {e}")
        return False

    # Load Chroma index
    try:
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            collection_name=COLLECTION_NAME,
            embedding_function=_embedder
        )
        count = _vectorstore._collection.count()
        print(f"Chroma loaded with {count} documents.")
        if count == 0:
            print("ERROR: Chroma is empty.")
            return False
    except Exception as e:
        print(f"Chroma load failed: {e}")
        return False

    _initialized = True
    return True


# ---------- MODELS ----------
class Query(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
    documents_found: int = 0
    response_time: float = 0.0
    status: str = "success"


# ---------- HELPER FUNCTIONS ----------
def is_relevant(current_query: str, previous_query: str):
    """
    Check if current question is related to previous conversation.
    Returns True for follow-up questions and context references.
    """
    current_lower = current_query.lower()
    previous_lower = previous_query.lower()
    
    # Check for pronouns that indicate follow-up
    follow_up_pronouns = ['he', 'she', 'it', 'they', 'his', 'her', 'their', 'that', 'this', 'there']
    current_words = set(current_lower.split())
    
    if any(pronoun in current_words for pronoun in follow_up_pronouns):
        return True
    
    # Check for question words that indicate follow-up
    question_words = ['what about', 'how about', 'and', 'also', 'tell me more']
    if any(word in current_lower for word in question_words):
        return True
    
    # Check for common keywords between questions
    current_keywords = set([w for w in current_lower.split() if len(w) > 3])
    previous_keywords = set([w for w in previous_lower.split() if len(w) > 3])
    
    if current_keywords and previous_keywords:
        common_words = current_keywords.intersection(previous_keywords)
        if len(common_words) >= 1:  # At least one significant word in common
            return True
    
    # Check if current question is asking about previous answer
    if 'previous' in current_lower or 'earlier' in current_lower or 'before' in current_lower:
        return True
    
    return False


def clean_answer(answer: str) -> str:
    """Remove file references and clean up answer text."""
    if not answer:
        return answer
    
    # Patterns to remove
    patterns_to_remove = [
        r'\.csv.*?(?=\s|$)',
        r'as per the record.*?(?=\s|$)',
        r'according to.*?(?=\s|$)',
        r'based on.*?(?=\s|$)',
        r'from the.*?(?=\s|$)',
        r'in the.*?file.*?(?=\s|$)',
        r'post_id.*?(?=\s|$)',
        r'title.*?(?=\s|$)',
        r'with.*?post_id.*?(?=\s|$)',
        r'with.*?title.*?(?=\s|$)',
        r'source_file.*?(?=\s|$)',
        r'unknown.*?(?=\s|$)',
    ]
    
    cleaned = answer.strip()
    
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Clean up whitespace and punctuation
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\s*,\s*,', ',', cleaned)
    cleaned = re.sub(r'\s*\.\s*\.', '.', cleaned)
    cleaned = cleaned.strip()
    
    # Fix sentence endings
    if cleaned.endswith(', .'):
        cleaned = cleaned[:-3] + '.'
    elif cleaned.endswith(','):
        cleaned = cleaned[:-1] + '.'
    elif cleaned and not cleaned.endswith(('.', '!', '?')):
        cleaned += '.'
    
    # Capitalize first letter
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    
    return cleaned


# ---------- ENHANCED SEARCH WITH HISTORY ----------
def search_documents(query: str, k=3):
    """
    Search with smart context from chat history.
    Checks history first (fast), then falls back to Chroma.
    """
    results = []
    
    # 1. FIRST: Check chat history for relevant previous conversations (ULTRA FAST)
    if _chat_history:
        for user_q, bot_a in reversed(_chat_history):  # Most recent first
            if is_relevant(query, user_q):
                # Create a pseudo-document from history
                hist_doc = Document(
                    page_content=f"Previous Q: {user_q}\nPrevious A: {bot_a}",
                    metadata={"source": "chat_history", "relevance": "high"}
                )
                results.append(hist_doc)
                logger.info(f"Found relevant history: {user_q[:50]}...")
                
                # If we found highly relevant history, prioritize it
                if len(results) >= 1 and ('he' in query.lower() or 'she' in query.lower() 
                                        or 'his' in query.lower() or 'her' in query.lower()):
                    # For pronoun-based follow-ups, history is usually enough
                    return results
    
    # 2. SECOND: Search Chroma for additional information
    remaining_slots = k - len(results)
    if remaining_slots > 0 and _vectorstore:
        try:
            db_results = _vectorstore.similarity_search(query, k=remaining_slots)
            results.extend(db_results)
            logger.info(f"Found {len(db_results)} documents from Chroma")
        except Exception as e:
            logger.error(f"Chroma search error: {e}")
    
    return results


# ---------- ENHANCED GENERATE RESPONSE WITH HISTORY ----------
def generate_response(query: str, docs: list):
    """Generate response with context from both documents and chat history."""
    if not docs:
        return "I couldn't find information about that in the available data.", []
    
    # Build context
    context_parts = []
    sources = []
    
    for doc in docs:
        src = doc.metadata.get("source_file", "Unknown")
        
        if src == "chat_history":
            # History entry - include as context
            context_parts.append(doc.page_content)
            if "chat_history" not in sources:
                sources.append("chat_history")
        else:
            # Regular document - clean it
            clean_content = doc.page_content
            # Remove any file references from content itself
            clean_content = re.sub(r'\.csv', '', clean_content)
            clean_content = re.sub(r'post_id\s+\d+', '', clean_content)
            clean_content = re.sub(r'title\s+\'.*?\'', '', clean_content)
            context_parts.append(clean_content)
            
            if src not in sources:
                sources.append(src)
    
    context = "\n---\n".join(context_parts)
    
    # Build history context string for prompt
    history_context = ""
    if _chat_history:
        history_context = "Recent conversation history:\n"
        for i, (user_q, bot_a) in enumerate(reversed(list(_chat_history)[-3:]), 1):
            history_context += f"{i}. User: {user_q}\n   Bot: {bot_a}\n"
    
    # Enhanced prompt with history awareness
    prompt = f"""
{history_context}

Current data context:
{context}

Question: {query}

Important Rules:
1. Answer using the information above
2. If referring to previous conversation, use that context
3. NEVER mention file names, sources, or metadata
4. If the information doesn't contain the answer, say: "The available data doesn't contain information about this."
5. Give a direct, factual answer

Answer (clean and direct):
"""
    
    try:
        ans = _llm.invoke(prompt).strip()
        cleaned_ans = clean_answer(ans)
        return cleaned_ans, sources
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "Sorry, I encountered an error while generating a response.", sources


# ---------- MAIN CHAT ROUTE + HYBRID APPROACH ----------
@app.post("/chat", response_model=ChatResponse)
async def chat(query: Query):
    start_time = time.time()

    if not _initialized:
        if not init_singleton():
            raise HTTPException(503, "Initialization failed")

    user_q = query.question.strip()
    user_q_lower = user_q.lower()
    logger.info(f"User asked: {user_q}")

    # INSTANT HYBRID GREETINGS (<1ms)
    greetings = {
        "hi": "Hi there! How can I assist you today?",
        "hii": "Hello! What can I do for you?",
        "hello": "Hello! How may I help you today?",
        "hey": "Hey! How can I support you?",
        "yo": "Yo! What's up? Need any help?",

        "good morning": "Good morning! Hope your day is off to a great start.",
        "good afternoon": "Good afternoon! How can I assist you?",
        "good evening": "Good evening! What would you like to know?",
        "good night": "Good night! Feel free to ask anything before you rest.",
        
        "how are you": "I'm doing great! Thanks for asking. How can I help you today?",
        "whats up": "All good here! How can I assist you today?",
        "what's up": "All good here! How can I assist you today?",
        "sup": "Hey! What do you need help with?",

        "thanks": "You're welcome! Happy to help anytime.",
        "thank you": "You're welcome! Let me know if you need anything else.",
        "ok": "Alright! Let me know if you need anything else.",
        "okay": "Okay! How else can I assist you?",
        "fine": "Good to hear! What can I do for you?",
        "hmm": "I'm here if you need anything.",
        
        "bye": "Goodbye! Take care. 😊",
        "good day": "Good day to you! How may I help?",
        "greetings": "Greetings! How can I assist you today?"
    }

    # Match if user message starts with or equals any hybrid phrase
    for phrase, reply in greetings.items():
        if user_q_lower.startswith(phrase):
            return ChatResponse(
                answer=reply,
                sources=[],
                documents_found=0,
                response_time=time.time() - start_time
            )

    # SMART SEARCH WITH HISTORY CONTEXT
    docs = search_documents(user_q, k=3)
    
    # Generate response
    answer, sources = generate_response(user_q, docs)
    
    # UPDATE CHAT HISTORY (store after successful response)
    _chat_history.append((user_q, answer))
    logger.info(f"Chat history updated. Total exchanges: {len(_chat_history)}")

    return ChatResponse(
        answer=answer,
        sources=sources,
        documents_found=len(docs),
        response_time=time.time() - start_time
    )


# ---------- DEBUG ENDPOINTS ----------
@app.get("/debug/history")
async def debug_history():
    """Debug endpoint to view chat history."""
    history_list = list(_chat_history)
    return {
        "history_count": len(history_list),
        "history": [
            {"user": user_q, "bot": bot_a, "index": i}
            for i, (user_q, bot_a) in enumerate(history_list, 1)
        ]
    }

@app.get("/debug/clear-history")
async def debug_clear_history():
    """Debug endpoint to clear chat history."""
    global _chat_history
    count = len(_chat_history)
    _chat_history.clear()
    return {"cleared": True, "messages_cleared": count}


# ---------- HOME ROUTE ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------- STARTUP ----------
@app.on_event("startup")
async def startup_event():
    print("\n" + "="*50)
    print(" SMART CHATBOT STARTING")
    print("="*50)
    print(f" LLM Model: {LLM_MODEL}")
    print(f" Chroma Directory: {CHROMA_DIR}")
    print(f" Collection: {COLLECTION_NAME}")
    print(" Chat History: Enabled")
    print("="*50)
    print("\n Debug Endpoints:")
    print("   http://localhost:8000/debug/history - View chat history")
    print("   http://localhost:8000/debug/clear-history - Clear history")
    print("="*50 + "\n")
    
    init_singleton()


# ---------- MAIN ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
