from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

from flask import Flask, jsonify, request

class Rag:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Rag, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Initialize only once
        if not hasattr(self, 'initialized'):
            self.initialized = True

            self.embedding = OpenAIEmbeddings()

            self.vector_store = Chroma(
                collection_name = "split_docs", 
                embedding_function = self.embedding,
                persist_directory = "./chroma.db")

            self.retriever = self.vector_store.as_retriever()

            self.llm = ChatOpenAI(model="gpt-4")
            self.prompt = hub.pull("rlm/rag-prompt")

            self.rag_chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )

app = Flask(__name__)

@app.route('/api/query/<string:question>', methods=['GET'])
def query(question):
    rag = Rag()
    answer = rag.rag_chain.invoke(question)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
