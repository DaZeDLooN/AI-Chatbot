import datetime
import os
import time

import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfReader


# Load API Key
def load_api_keys():
    load_dotenv()
    huggingface_token = os.getenv("HUGGINGFACE_API_TOKEN")
    return huggingface_token

# Read PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Scrape web page content
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = "\n".join([para.get_text() for para in paragraphs])
    return text

# Split into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    chunks = text_splitter.split_text(text)
    return chunks

# Save into VectorDB
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# HuggingFace Zephyr Answering
def hf_answer(hf_client, prompt, model_name):
    try:
        output = hf_client.text_generation(
            prompt,
            model=model_name,
            max_new_tokens=400,
            temperature=0.3,
        )
        if hasattr(output, 'generated_text'):
            return output.generated_text
        else:
            return output
    except Exception as e:
        st.error(f"Error contacting HuggingFace API: {e}")
        return "Error generating answer."

# Handle user questions
def user_input(user_question, chat_history, model_name, hf_client):
    time.sleep(2)

    if 'last_user_question' not in st.session_state or st.session_state.last_user_question != user_question:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question, k=2)

        context = "\n".join(doc.page_content for doc in docs)

        full_prompt = f"""Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{user_question}\nAnswer:"""

        response_text = hf_answer(hf_client, full_prompt, model_name)

        st.session_state.last_user_question = user_question
        st.session_state.response_text = response_text
        st.session_state.docs = docs

    response_text = st.session_state.response_text
    docs = st.session_state.docs

    st.markdown("**Question:** " + user_question)
    st.markdown("**Answer:** " + response_text)

    with st.expander("Show Source(s)"):
        for doc in docs:
            st.markdown(doc.page_content)

    feedback = st.radio("Was this answer helpful?", ("Good 👍", "Bad 👎"), index=None)

    if feedback == "Bad 👎":
        refinement = st.text_input("Suggest how to improve the answer (e.g., explain more, add examples, simplify):")
        if refinement:
            refined_prompt = f"""Refine the following answer based on this feedback: \"{refinement}\"\n\nOriginal Question: {user_question}\nOriginal Answer: {response_text}\n\nRefined Answer:"""
            refined_response = hf_answer(hf_client, refined_prompt, model_name)
            st.markdown("**Refined Answer:**")
            st.markdown(refined_response)
            chat_history.append((user_question, refined_response))
    else:
        chat_history.append((user_question, response_text))

# Summarize Content
def summarize_content(raw_text, hf_client, model_name):
    prompt = f"""
    Summarize the following text into 10 bullet points:

    {raw_text}
    """
    response = hf_answer(hf_client, prompt, model_name)
    return response

# Generate Potential FAQs
def generate_potential_faqs(raw_text, hf_client, model_name):
    prompt = f"""
    Based on the following text, list 10 potential Frequently Asked Questions (FAQs):

    {raw_text}
    """
    response = hf_answer(hf_client, prompt, model_name)
    return response

# Main App
def main():
    huggingface_token = load_api_keys()
    hf_client = InferenceClient(token=huggingface_token)

    st.set_page_config(page_title="📚 AI Powered FAQ Chatbot", layout="wide")
    st.title("📚 AI Powered FAQ Chatbot")
    st.markdown("""---""")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.title("Settings")
        usage_mode = st.selectbox("Choose Usage Mode", ("Static Mode (Upload File / Text)", "Dynamic Mode (Scrape Website)"))
        model_choice = st.selectbox("Select Model", ("Zephyr-7B-β",))

        if model_choice == "Zephyr-7B-β":
            model_name = "HuggingFaceH4/zephyr-7b-beta"

        if usage_mode == "Static Mode (Upload File / Text)":
            content_option = st.radio("Choose Content Source", ("Upload PDF", "Paste Static Text"))

            if content_option == "Upload PDF":
                pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
                if st.button("Submit & Process") and pdf_docs:
                    with st.spinner("Processing PDFs..."):
                        raw_text = get_pdf_text(pdf_docs)
                        st.session_state.raw_text = raw_text
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("PDFs processed and saved!")

            else:
                static_text = st.text_area("Paste your static text content here")
                if st.button("Submit & Process Text") and static_text:
                    with st.spinner("Processing Text..."):
                        st.session_state.raw_text = static_text
                        text_chunks = get_text_chunks(static_text)
                        get_vector_store(text_chunks)
                        st.success("Static text processed and saved!")

        else:
            url = st.text_input("Enter Website URL to Scrape")
            if st.button("Scrape and Process") and url:
                with st.spinner("Scraping website..."):
                    raw_text = scrape_website(url)
                    st.session_state.raw_text = raw_text
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Website scraped and saved!")

        if "raw_text" in st.session_state:
            with st.expander("Preview Uploaded/Scraped Content"):
                st.write(st.session_state.raw_text[:3000] + "...")

            if st.button("Summarize Content"):
                with st.spinner("Summarizing..."):
                    summary = summarize_content(st.session_state.raw_text, hf_client, model_name)
                    st.subheader("Summary:")
                    st.markdown(summary)

            if st.button("Generate Potential FAQ Questions"):
                with st.spinner("Generating FAQs..."):
                    faqs = generate_potential_faqs(st.session_state.raw_text, hf_client, model_name)
                    st.subheader("Potential FAQ Questions:")
                    st.markdown(faqs)

    st.subheader("Ask a Question about your content:")
    user_question = st.text_input("Type your question here")
    if user_question:
        with st.spinner("Thinking..."):
            user_input(user_question, st.session_state.chat_history, model_name, hf_client)

    if st.session_state.chat_history:
        st.subheader("Chat History")
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"**{i+1}. Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")

    if st.session_state.chat_history:
        chat_log = ""
        for q, a in st.session_state.chat_history:
            chat_log += f"Q: {q}\nA: {a}\n---\n"

        st.download_button("Download Chat Log", chat_log, file_name=f"chat_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt")

if __name__ == "__main__":
    main()