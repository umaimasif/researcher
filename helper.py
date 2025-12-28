import os
import json
import re
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# State Definition
class State(TypedDict):
    query: str
    search_results: list
    urls: List[str]
    db: object
    summaries: str
    newsletter: str

# Initialize Components
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("GOOGLE_API_KEY"), 
    model="gemini-1.5-flash", 
    temperature=0
)

# Nodes
def search_tavily_node(state: State):
    tool = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
    response = tool.invoke({"query": state["query"]})
    return {"search_results": response}

def pick_best_articles_node(state: State):
    response_str = json.dumps(state["search_results"])
    prompt = f"QUERY RESPONSE: {response_str}\nReturn ONLY a raw JSON array of the top 3 URLs for: {state['query']}"
    urls_response = llm.invoke(prompt)
    clean_json = re.sub(r"```json|```", "", urls_response.content).strip()
    try:
        url_list = json.loads(clean_json)
    except:
        url_list = ["https://www.google.com"]
    return {"urls": url_list}

def extract_content_node(state: State):
    loader = WebBaseLoader(state["urls"], header_template={'User-Agent': 'Mozilla/5.0'})
    docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
    return {"db": FAISS.from_documents(docs, embeddings)}

def summarizer_node(state: State):
    docs = state["db"].similarity_search(state["query"], k=4)
    content = " ".join([d.page_content for d in docs])
    summary = llm.invoke(f"Summarize as 5 bullet points:\n\n{content}")
    return {"summaries": summary.content}

def generate_newsletter_node(state: State):
    prompt = f"Write an informal newsletter using this: {state['summaries']}. Start with 'Hi All!' and end with 'Umaima Asif - Learner'"
    newsletter = llm.invoke(prompt)
    return {"newsletter": newsletter.content}

# Compilation Function
def compile_workflow():
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
    return workflow.compile()
