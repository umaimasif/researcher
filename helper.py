import os
import json
import re
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -----------------------------
# Environment Setup
# -----------------------------
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Embeddings and LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# FIX: Model name changed to gemini-1.5-flash
llm = ChatGoogleGenerativeAI(
    api_key=google_api_key, 
    model="gemini-1.5-flash", 
    temperature=0,
    convert_system_message_to_human=True
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
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    tool = TavilySearchResults(api_key=tavily_api_key)
    response_json = tool.invoke({"query": state["query"], "k": 5})
    return {"search_results": response_json}

def pick_best_articles_node(state: State):
    response_str = json.dumps(state["search_results"])
    messages = [
        {"role": "system", "content": "You are a world-class journalist, researcher, and developer."},
        {"role": "user", "content": f"QUERY RESPONSE: {response_str}\n\nChoose the best 3 articles from the list and return ONLY a raw JSON array of URLs for {state['query']}. No markdown."}
    ]
    urls_response = llm.invoke(messages)
    
    # FIX: Clean Markdown formatting if Gemini adds it
    clean_content = re.sub(r"```json|```", "", urls_response.content).strip()
    
    try:
        url_list = json.loads(clean_content)
    except:
        url_list = ["https://www.google.com"]
    
    return {"urls": url_list}

def extract_content_node(state: State):
    # FIX: Added User-Agent to prevent 403 Forbidden errors
    loader = WebBaseLoader(
        state["urls"], 
        header_template={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    )
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    db = FAISS.from_documents(docs, embeddings)
    return {"db": db}

def summarizer_node(state: State):
    docs = state["db"].similarity_search(state["query"], k=4)
    docs_page_content = " ".join([d.page_content for d in docs])
    messages = [
        {"role": "system", "content": "You are a professional newsletter writer."},
        {"role": "user", "content": f"{docs_page_content}\n\nSummarize the above text into a short newsletter about {state['query']}. Format it like Tim Ferriss' '5-Bullet Friday'."}
    ]
    summary_response = llm.invoke(messages)
    return {"summaries": summary_response.content.strip()}

def generate_newsletter_node(state: State):
    messages = [
        {"role": "system", "content": "You are an engaging and informal newsletter writer."},
        {"role": "user", "content": f"{state['summaries']}\n\nWrite a newsletter about {state['query']} in an engaging and informal style, starting with:\nHi All!\nHere is your weekly dose of the Tech Newsletter...\n\nEnd with:\nUmaima Asif\n- Learner"}
    ]
    newsletter_response = llm.invoke(messages)
    return {"newsletter": newsletter_response.content.strip()}

# -----------------------------
# Build and Compile Graph
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
