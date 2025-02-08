from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from glob import glob
import sys

pdf_files = glob('../datasets/fancy-machine-manuals/en/*.pdf')
loaders = [PyPDFLoader(pdf) for pdf in pdf_files]

documents = []
for file in loaders:
    documents.extend(file.load())

splitter = RecursiveCharacterTextSplitter(chunk_size = 150, chunk_overlap = 75)

embedding = OpenAIEmbeddings()

vector_store = Chroma(
    collection_name = "split_docs", 
    embedding_function = embedding,
    persist_directory = "./chroma.db")

retriever = vector_store.as_retriever()

retriever.add_documents(documents)

sys.exit(0)