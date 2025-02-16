from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from glob import glob
import argparse

pdf_files = glob('../datasets/fancy-machine-manuals/en/*.pdf')
loaders = [PyPDFLoader(pdf) for pdf in pdf_files]

documents = []
for file in loaders:
    documents.extend(file.load())

parser = argparse.ArgumentParser(description='Script to build the vector storage knowledgement base')
parser.add_argument('-em', '--emodel', type=str, required=False, default='text-embedding-ada-002', help='Embeddong model name to use')
args = parser.parse_args()

embedding = OpenAIEmbeddings(model=args.emodel)

vector_store = Chroma(
    collection_name = "split_docs", 
    embedding_function = embedding,
    persist_directory = "./chroma.db")

splitter = RecursiveCharacterTextSplitter(chunk_size = 150, chunk_overlap = 75)
splits = splitter.split_documents(documents)
vector_store.add_documents(splits)
