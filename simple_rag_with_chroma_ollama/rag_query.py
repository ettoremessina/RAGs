from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
import sys

def query(question):
    ollama_url = "http://localhost:11434"
    model_name = "llama3" # "gemma2" or others available in your ollama service

    embedding = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_url)

    vector_store = Chroma(
        collection_name = "split_docs", 
        embedding_function = embedding,
        persist_directory = "./chroma.db")

    splitter = RecursiveCharacterTextSplitter(chunk_size = 150, chunk_overlap = 75)

    retriever = vector_store.as_retriever()

    llm = Ollama(model = model_name, base_url=ollama_url)
    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)

if len(sys.argv) < 2:
    print("Error: Please provide the question as command line argument", file=sys.stderr)
    print("Usage: python rag_querty \"your question\"", file=sys.stderr)
    sys.exit(1)

question = " ".join(sys.argv[1:])
answer = query(question)
print(answer)

sys.exit(0)


