import os
import json
import re
from dotenv import load_dotenv

# Updated Imports
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
except ImportError:
    from langchain_tavily import TavilySearchResults

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# Standardizing the text splitter import
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# Fix for identification
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# LLM setup using Groq
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
# Nodes (Steps)
# -----------------------------
def search_tavily_node(state: State):
    # This initialization is more robust for current LangChain versions
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    tool = TavilySearchResults(api_key=tavily_api_key, k=5)
    
    # Use the tool to search
    response = tool.invoke(state["query"])
    return {"search_results": response}

def pick_best_articles_node(state: State):
    # Groq handles JSON extraction very well
    response_str = json.dumps(state["search_results"])
    prompt = f"QUERY RESPONSE: {response_str}\n\nReturn ONLY a raw JSON array of the top 3 URLs for: {state['query']}. Example: [\"https://url1.com\", \"https://url2.com\"]"
    
    urls_response = llm.invoke(prompt)
    clean_content = re.sub(r"```json|```", "", urls_response.content).strip()
    
    try:
        url_list = json.loads(clean_content)
        # Ensure it's a list
        if not isinstance(url_list, list):
            url_list = [url_list]
    except:
        url_list = ["https://www.google.com"]
    return {"urls": url_list[:3]}

def extract_content_node(state: State):
    loader = WebBaseLoader(
        state["urls"],
        header_template={'User-Agent': os.environ["USER_AGENT"]}
    )
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    
    # Create the vector store
    db = FAISS.from_documents(docs, embeddings)
    return {"db": db}

def summarizer_node(state: State):
    # Get relevant docs from FAISS
    docs = state["db"].similarity_search(state["query"], k=4)
    docs_page_content = " ".join([d.page_content for d in docs])
    
    prompt = f"Context: {docs_page_content}\n\nSummarize into a 5-bullet newsletter about {state['query']}."
    summary_response = llm.invoke(prompt)
    return {"summaries": summary_response.content.strip()}

def generate_newsletter_node(state: State):
    prompt = f"Summary: {state['summaries']}\n\nWrite an engaging newsletter about {state['query']}. Start with 'Hi All! Here is your weekly dose...' and end with 'Umaima Asif - Learner'"
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
