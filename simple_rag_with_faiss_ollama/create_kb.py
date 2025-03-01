from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from glob import glob
import argparse

pdf_files = glob('../datasets/fancy-machine-manuals/en/*.pdf')
loaders = [PyPDFLoader(pdf) for pdf in pdf_files]

documents = []
for file in loaders:
    documents.extend(file.load())

parser = argparse.ArgumentParser(description='Script to build the vector storage knowledgement base')
parser.add_argument('-em', '--emodel', type=str, required=False, default='nomic-embed-text', help='Embedding model name to use')
parser.add_argument('-ou', '--ollamaurl', type=str, required=False, default='http://localhost:11434', help='Ollama url')
args = parser.parse_args()

embedding = OllamaEmbeddings(model=args.emodel, base_url=args.ollamaurl)

splitter = RecursiveCharacterTextSplitter(chunk_size = 150, chunk_overlap = 75)
splits = splitter.split_documents(documents)

faiss_index = FAISS.from_documents(
        splits,
        embedding
)
faiss_index.save_local("faiss.db")
