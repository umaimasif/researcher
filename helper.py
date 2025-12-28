import os
import json
import re
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq  # Changed from Google to Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# Fix for the import error you saw earlier
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# Identification for web requests
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# FIX: Switch to Groq (using Llama 3 for high speed and accuracy)
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# -----------------------------
# State Definition
# -----------------------------
class State(TypedDict):
    query: str
    search_results: list 
    urls: List[str]
    db: object
    summaries: str
    newsletter: str

# -----------------------------
# Nodes (Steps) - Logic stays the same, just better reliability
# -----------------------------
def search_tavily_node(state: State):
    tool = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
    response_json = tool.invoke({"query": state["query"], "k": 5})
    return {"search_results": response_json}

def pick_best_articles_node(state: State):
    response_str = json.dumps(state["search_results"])
    prompt = f"QUERY RESPONSE: {response_str}\n\nReturn ONLY a raw JSON array of the top 3 URLs for: {state['query']}"
    
    urls_response = llm.invoke(prompt)
    clean_content = re.sub(r"```json|```", "", urls_response.content).strip()
    
    try:
        url_list = json.loads(clean_content)
    except:
        url_list = ["https://www.google.com"]
    return {"urls": url_list}

def extract_content_node(state: State):
    loader = WebBaseLoader(state["urls"])
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    db = FAISS.from_documents(docs, embeddings)
    return {"db": db}

def summarizer_node(state: State):
    docs = state["db"].similarity_search(state["query"], k=4)
    docs_page_content = " ".join([d.page_content for d in docs])
    prompt = f"{docs_page_content}\n\nSummarize into a short newsletter about {state['query']} in '5-Bullet Friday' style."
    summary_response = llm.invoke(prompt)
    return {"summaries": summary_response.content.strip()}

def generate_newsletter_node(state: State):
    prompt = f"{state['summaries']}\n\nWrite a newsletter about {state['query']} starting with 'Hi All! Here is your weekly dose...' and ending with 'Umaima Asif - Learner'"
    newsletter_response = llm.invoke(prompt)
    return {"newsletter": newsletter_response.content.strip()}

# -----------------------------
# Graph Construction
# -----------------------------
workflow = StateGraph(State)
workflow.add_node("search_tavily", search_tavily_node)
workflow.add_node("pick_best_articles", pick_best_articles_node)
workflow.add_node("extract_content", extract_content_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("generate_newsletter", generate_newsletter_node)

workflow.set_entry_point("search_tavily")
workflow.add_edge("search_tavily", "pick_best_articles")
workflow.add_edge("pick_best_articles", "extract_content")
workflow.add_edge("extract_content", "summarizer")
workflow.add_edge("summarizer", "generate_newsletter")
workflow.add_edge("generate_newsletter", END)

app = workflow.compile()
