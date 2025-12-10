import streamlit as st
import requests
import json

st.set_page_config(page_title="IPCC RAG Demo", page_icon="🌍", layout="wide")

st.title("🌍 RAG Demo - IPCC AR6 Climate Reports")
st.markdown("Ask questions about climate change based on IPCC AR6 reports")

API_URL = "http://localhost:8000/ask"

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ask a Question")
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What are the main causes of climate change?",
        label_visibility="collapsed"
    )
    
    ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)

with col2:
    st.subheader("Example Questions")
    st.markdown("""
    - What are the main causes of climate change?
    - What is the current global temperature increase?
    - What are the projected impacts of climate change?
    - What mitigation strategies does IPCC recommend?
    """)

if ask_button and question:
    with st.spinner("🤔 Thinking..."):
        try:
            response = requests.post(
                API_URL,
                json={"question": question},
                timeout=30
            )
            
            if response.ok:
                data = response.json()
                
                st.subheader("📝 Answer")
                st.write(data["answer"])
                
                st.subheader("📚 Sources")
                sources = data.get("sources", [])
                
                if sources:
                    for i, source in enumerate(sources, 1):
                        with st.expander(f"Source {i}"):
                            st.markdown(f"**Content Preview:**")
                            st.text(source.get("content", "No content available"))
                            
                            st.markdown(f"**Metadata:**")
                            metadata = source.get("metadata", {})
                            st.json(metadata)
                else:
                    st.info("No sources returned")
            else:
                st.error(f"API Error: {response.status_code}")
                st.text(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the API. Make sure the FastAPI server is running on port 8000!")
            st.code("uvicorn app:app --reload --port 8000")
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The question might be taking too long to process.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

elif ask_button and not question:
    st.warning("⚠️ Please enter a question first!")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This demo uses:
    - **Ollama** for local LLMs
    - **LangChain** for RAG pipeline
    - **Chroma** for vector storage
    - **FastAPI** for the backend
    - **Streamlit** for the UI
    
    Data source: IPCC AR6 Reports
    """)
    
    st.header("🔧 Status")
    
    try:
        health = requests.get("http://localhost:8000/", timeout=2)
        if health.ok:
            st.success("✅ API is running")
        else:
            st.error("❌ API error")
    except:
        st.error("❌ API not reachable")
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Ollama + LangChain")