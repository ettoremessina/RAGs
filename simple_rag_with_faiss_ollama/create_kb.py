from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from glob import glob

ollama_url = "http://localhost:11434"

pdf_files = glob('../datasets/fancy-machine-manuals/en/*.pdf')
loaders = [PyPDFLoader(pdf) for pdf in pdf_files]

documents = []
for file in loaders:
    documents.extend(file.load())

splitter = RecursiveCharacterTextSplitter(chunk_size = 150, chunk_overlap = 75)
splits = splitter.split_documents(documents)

embedding = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_url)

faiss_index = FAISS.from_documents(
        splits,
        embedding
)
faiss_index.save_local("faiss.db")
