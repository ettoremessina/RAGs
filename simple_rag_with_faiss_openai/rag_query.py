from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
import sys

def query(question):
    embedding = OpenAIEmbeddings()

    vector_store = FAISS.load_local(
        "faiss.db",
        embedding,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k": 3}
    )

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


