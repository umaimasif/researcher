import os
import json
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
try:
    # Newer versions
    from langchain.text_splitter import RecursiveCharacterTextSplitter as CharacterTextSplitter
except ModuleNotFoundError:
    # Fallback for older versions
    from langchain.text_splitter import CharacterTextSplitter

# -----------------------------
# Environment Setup
# -----------------------------
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Embeddings and LLM
embeddings = GoogleGenerativeAIEmbeddings(api_key=google_api_key, model="models/embedding-001")
llm = ChatGoogleGenerativeAI(api_key=google_api_key, model="models/gemini-2.5-flash-lite", temperature=0,convert_system_message_to_human=True)

# -----------------------------
# State Definition
# -----------------------------
class State(TypedDict):
    query: str
    search_results: dict
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
        {
            "role": "system",
            "content": "You are a world-class journalist, researcher, and developer."
        },
        {
            "role": "user",
            "content": f"""
QUERY RESPONSE: {response_str}

Above is the list of search results for the query {state['query']}.

Please choose the best 3 articles from the list and return ONLY an array of URLs.
Do not include anything else.
If invalid, return ["https://www.google.com"].
"""
        }
    ]
    
    urls_response = llm.invoke(messages)
    
    try:
        url_list = json.loads(urls_response.content)
    except:
        url_list = ["https://www.google.com"]
    
    print("✅ Picked URLs:", url_list)
    return {"urls": url_list}

def extract_content_node(state: State):
    loader = UnstructuredURLLoader(urls=state["urls"])
    data = loader.load()
    
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    
    docs = text_splitter.split_documents(data)
    db = FAISS.from_documents(docs, embeddings)
    return {"db": db}

def summarizer_node(state: State):
    docs = state["db"].similarity_search(state["query"], k=4)
    docs_page_content = " ".join([d.page_content for d in docs])
    
    messages = [
        {
            "role": "system",
            "content": "You are a professional newsletter writer."
        },
        {
            "role": "user",
            "content": f"""
{docs_page_content}

Summarize the above text into a short newsletter about {state['query']}.
Format it like Tim Ferriss' "5-Bullet Friday".
"""
        }
    ]
    
    summary_response = llm.invoke(messages)
    return {"summaries": summary_response.content.strip()}

def generate_newsletter_node(state: State):
    messages = [
        {
            "role": "system",
            "content": "You are an engaging and informal newsletter writer."
        },
        {
            "role": "user",
            "content": f"""
{state['summaries']}

Write a newsletter about {state['query']} in an engaging and informal style, starting with:
Hi All!
Here is your weekly dose of the Tech Newsletter...

End with:
Umaima Asif
- Learner
"""
        }
    ]
    
    newsletter_response = llm.invoke(messages)
    return {"newsletter": newsletter_response.content.strip()}

# -----------------------------
# Build the LangGraph
# -----------------------------
workflow = StateGraph(State)

workflow.add_node("search_tavily", search_tavily_node)
workflow.add_node("pick_best_articles", pick_best_articles_node)
workflow.add_node("extract_content", extract_content_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("generate_newsletter", generate_newsletter_node)

# Define edges
workflow.set_entry_point("search_tavily")
workflow.add_edge("search_tavily", "pick_best_articles")
workflow.add_edge("pick_best_articles", "extract_content")
workflow.add_edge("extract_content", "summarizer")
workflow.add_edge("summarizer", "generate_newsletter")
workflow.add_edge("generate_newsletter", END)

# Compile
app = workflow.compile()

# -----------------------------
# Run the Graph
# -----------------------------
if __name__ == "__main__":
    query = "Latest AI breakthroughs in healthcare"
    final_state = app.invoke({"query": query})
    print("\n📩 Final Newsletter:\n", final_state["newsletter"])




