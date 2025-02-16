from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain import hub
import argparse
from flask import Flask, jsonify, request

parser = argparse.ArgumentParser(description='API service to chat on knowledgement base using RAG technology')
parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
parser.add_argument('-em', '--emodel', type=str, required=False, default='text-embedding-ada-002', help='Embeddong model name to use (default: nomic-embed-text)')
parser.add_argument('-m', '--model', type=str, required=False, default='gpt-4', help='Model name to use (e.g. gpt-4, gpt-4o-mini, ...; default: llama3.2)')
args = parser.parse_args()

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

            self.embedding = OpenAIEmbeddings(model=args.emodel)

            self.vector_store = Chroma(
                collection_name = "split_docs", 
                embedding_function = self.embedding,
                persist_directory = "./chroma.db")

            self.retriever = self.vector_store.as_retriever()

            self.llm = ChatOpenAI(model=args.model, temperature=0.)
            self.prompt = hub.pull("rlm/rag-prompt")

            self.rag_chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )

            self.config = {}
            if args.verbose:
                self.config = self.config | {'callbacks': [ConsoleCallbackHandler()]}

app = Flask(__name__)

@app.route('/api/rag/<string:question>', methods=['GET'])
def query(question):
    rag = Rag()
    answer = rag.rag_chain.invoke(
        question,
        config=rag.config)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
