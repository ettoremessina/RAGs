from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from glob import glob

ollama_url = "http://localhost:11434"

pdf_files = glob('../docs/en/*.pdf')
loaders = [PyPDFLoader(pdf) for pdf in pdf_files]

documents = []
for file in loaders:
    documents.extend(file.load())

splitter = RecursiveCharacterTextSplitter(chunk_size = 150, chunk_overlap = 75)

embedding = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_url)

vector_store = Chroma(
    collection_name = "split_docs", 
    embedding_function = embedding,
    persist_directory = "./chroma.db")

retriever = vector_store.as_retriever()

retriever.add_documents(documents)

