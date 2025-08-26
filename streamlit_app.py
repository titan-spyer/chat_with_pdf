import asyncio
from dotenv import load_dotenv
import os
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def main():
    # In some environments, especially with libraries like langchain that use asyncio,
    # a RuntimeError can occur if there's no current event loop.
    # This code ensures that an event loop is available for the current thread.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    load_dotenv()
    # Add API key check
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY environment variable not set. Please create a .env file and add it.")
        st.info("You can get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).")
        st.stop()
    
    st.set_page_config(page_title="PDF Chatbot", page_icon=":books:")
    st.header("PDF Chatbot with Google Generative AI :books:")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    with st.sidebar:
        st.subheader("Your document")
        pdf = st.file_uploader("Upload your PDF and click 'Process'", type=["pdf"])
        if st.button("Process"):
            if pdf is not None:
                with st.spinner("Processing PDF..."):
                    # Extract text from PDF
                    pdf_reader = PdfReader(pdf)
                    text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
                    
                    # Split text into chunks
                    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_text(text)

                    # Create embeddings and vector store
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    knowledge_base = FAISS.from_texts(chunks, embeddings)

                    # Create the conversation chain
                    # Use gemini-pro for more generous free-tier rate limits
                    llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
                    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
                    st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=knowledge_base.as_retriever(),
                        memory=memory
                    )
                    st.success("PDF processed! You can now ask questions about it.")
            else:
                st.warning("Please upload a PDF file first.")

    if st.session_state.conversation:
        user_query = st.text_input("Ask about your PDF:")
        if user_query and st.session_state.conversation:
            with st.spinner("Thinking..."):
                st.session_state.conversation.invoke({'question': user_query})
                st.session_state.chat_history = st.session_state.conversation.memory.chat_memory.messages

        # Display chat history from session state
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(f"**You:** {message.content}")
                else:
                    st.write(f"**Bot:** {message.content}")

if __name__ == '__main__':
    main()