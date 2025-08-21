import os
import json
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# -----------------------------
# Environment Setup
# -----------------------------
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

# Embeddings and LLM
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

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
    search = TavilySearchResults(k=5)
    response_json = search.run(state["query"])
    print("🔍 Search Results:", response_json)
    return {"search_results": response_json}


def pick_best_articles_node(state: State):
    response_str = json.dumps(state["search_results"])
    template = """ 
      You are a world class journalist, researcher, and developer.
      
      QUERY RESPONSE:{response_str}
      
      Above is the list of search results for the query {query}.
      
      Please choose the best 3 articles from the list and return ONLY an array of the urls.  
      Do not include anything else. 
      If invalid, return ["https://www.google.com"].
    """
    prompt_template = PromptTemplate(
        input_variables=["response_str", "query"],
        template=template
    )
    urls = llm.invoke(prompt_template.format(response_str=response_str, query=state["query"]))
    try:
        url_list = json.loads(urls.content)
    except:
        url_list = ["https://www.google.com"]
    print("✅ Picked URLs:", url_list)
    return {"urls": url_list}


from langchain_community.vectorstores import Chroma

def extract_content_node(state: State):
    loader = UnstructuredURLLoader(urls=state["urls"])
    data = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    docs = text_splitter.split_documents(data)
    db = Chroma.from_documents(docs, embeddings)  
    return {"db": db}


def summarizer_node(state: State):
    docs = state["db"].similarity_search(state["query"], k=4)
    docs_page_content = " ".join([d.page_content for d in docs])

    template = """
       {docs}
        Summarize the above text into a short newsletter about {query}.
        Format it like Tim Ferriss' "5-Bullet Friday".
    """
    prompt_template = PromptTemplate(input_variables=["docs", "query"], template=template)
    summary = llm.invoke(prompt_template.format(docs=docs_page_content, query=state["query"]))
    return {"summaries": summary.content.strip()}


def generate_newsletter_node(state: State):
    template = """
    {summaries_str}
        Write a newsletter about {query} in an engaging and informal style,
        starting with:
        "Hi All!
         Here is your weekly dose of the Tech Newsletter..."
         
        End with:
        Umaima Asif
        - Learner and Teacher
    """
    prompt_template = PromptTemplate(
        input_variables=["summaries_str", "query"], template=template
    )
    newsletter = llm.invoke(
        prompt_template.format(summaries_str=state["summaries"], query=state["query"])
    )
    return {"newsletter": newsletter.content.strip()}

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
