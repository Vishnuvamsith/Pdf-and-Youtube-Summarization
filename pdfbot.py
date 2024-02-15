from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
def loader(pdfs):
    documents = []
    for pdf in pdfs:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, pdf.name)
        with open(file_path, "wb") as f:
            f.write(pdf.read())
        loader = PDFPlumberLoader(file_path=file_path)
        doc = loader.load()
        documents += doc
        os.remove(file_path)
        os.rmdir(temp_dir)
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    texts=text_splitter.split_documents(documents)
    return texts
def embed(texts):
    persist_directory='db'
    embeddings=OpenAIEmbeddings()
    vectordb=Chroma.from_documents(texts,embeddings,persist_directory=persist_directory)
    vectordb.persist()
    vectordb=None
    return persist_directory,embeddings
def predict(query,persist_directory,embeddings):
    vectordb=Chroma(persist_directory=persist_directory,embedding_function=embeddings)
    retriever=vectordb.as_retriever(search_type="similarity",search_kwargs=({"k":4}))
    rqa=RetrievalQA.from_chain_type(llm=OpenAI(),chain_type="stuff",retriever=retriever,return_source_documents=True)
    result=rqa(query)
    return result['result']
st.title("Chat With Your PDF's")
uploaded_files = st.file_uploader("Upload one or more PDF documents", type="pdf", accept_multiple_files=True)
query = st.text_input("Ask your question")
if st.button("Generate Answer"):
    if uploaded_files:
        texts = loader(uploaded_files)
        if(texts):
            persist_directory,embeddings=embed(texts)
            output=predict(query,persist_directory,embeddings)
            st.subheader("Answer:")
            st.write(output)
    else:
        st.warning("Please upload one or more PDF documents.")
