import os
import streamlit as st
from dotenv import load_dotenv
from helper import app  # import the compiled LangGraph workflow

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(
        page_title="Researcher...",
        page_icon=":parrot:",
        layout="wide"
    )
    
    st.header("Search, Discover & Explore Anything 🔍✨")
    query = st.text_input("Enter a topic...")

    if query:
        with st.spinner(f"Searching for '{query}'..."):
            # Run the LangGraph workflow
            try:
                final_state = app.invoke({"query": query})
            except Exception as e:
                st.error(f"Error running workflow: {e}")
                return

            # Extract results
            search_results = final_state.get("search_results", {})
            urls = final_state.get("urls", [])
            summaries = final_state.get("summaries", "")
            newsletter_thread = final_state.get("newsletter", "")

            # Display results in Streamlit expanders
            with st.expander("Search Results"):
                st.json(search_results)
            
            with st.expander("Best URLs"):
                st.write(urls)
            
            with st.expander("Summaries"):
                st.write(summaries)
            
            with st.expander("Newsletter"):
                st.write(newsletter_thread)

        st.success("Done! ✅")

if __name__ == '__main__':
    main()


