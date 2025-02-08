from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
import sys

def query(question):
    embedding = OpenAIEmbeddings()

    vector_store = Chroma(
        collection_name = "split_docs", 
        embedding_function = embedding,
        persist_directory = "./chroma.db")

    retriever = vector_store.as_retriever()

    llm = ChatOpenAI(model="gpt-4")
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


