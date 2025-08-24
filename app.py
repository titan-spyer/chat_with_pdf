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
    load_dotenv()
    # Add API key check
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY environment variable not set. Please create a .env file and add it.")
        st.info("You can get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).")
        st.stop()
    
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    st.set_page_config(page_title="PDF Chatbot", page_icon=":books:")
    st.header("PDF Chatbot with Google Generative AI :books:")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    pdf = st.file_uploader("Upload your PDF", type=["pdf"])

    if pdf is not None:
        # To avoid reprocessing on every interaction, we process the PDF only once
        if st.session_state.conversation is None:
            with st.spinner("Processing PDF..."):
                pdf_reader = PdfReader(pdf)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
                text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
                chunks = text_splitter.split_text(text)

                # FIX: The 'model' parameter is required for GoogleGenerativeAIEmbeddings.
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                knowledge_base = FAISS.from_texts(chunks, embeddings)

                # Create the conversation chain
                llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.3)
                memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
                st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=knowledge_base.as_retriever(),
                    memory=memory
                )
                st.success("PDF processed! You can now ask questions.")

        user_query = st.text_input("Ask about your PDF:")
        if user_query and st.session_state.conversation:
            response = st.session_state.conversation({'question': user_query})
            st.session_state.chat_history = response['chat_history']

            # Display chat history
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(f"**You:** {message.content}")
                else:
                    st.write(f"**Bot:** {message.content}")

if __name__ == '__main__':
    main()