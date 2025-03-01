from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain import hub
import argparse

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

parser = argparse.ArgumentParser(description='Script to chat on knowledgement base using RAG technology')
parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
parser.add_argument('-em', '--emodel', type=str, required=False, default='nomic-embed-text', help='Embedding model name to use (default: nomic-embed-text)')
parser.add_argument('-m', '--model', type=str, required=False, default='llama3.2', help='Model name to use (e.g. gemma2, llama3.2, ...; default: llama3.2)')
parser.add_argument('-ou', '--ollamaurl', type=str, required=False, default='http://localhost:11434', help='Ollama url')
args = parser.parse_args()

embedding = OllamaEmbeddings(model=args.emodel, base_url=args.ollamaurl)

vector_store = FAISS.load_local(
    "faiss.db",
    embedding,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 3}
)

llm = Ollama(model=args.model, base_url=args.ollamaurl)

prompt = hub.pull("rlm/rag-prompt")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

config = {}
if args.verbose:
    config = config | {'callbacks': [ConsoleCallbackHandler()]}

print("Chat with me (ctrl+D to quit)!\n")

while True:
    try:
        question = input("human: ")
        answer = rag_chain.invoke(
            question,
            config=config
        )
        print("bot  : ", answer)
    except EOFError:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"{bcolors.FAIL}{type(e)}")
        print(f"{bcolors.FAIL}{e.args}")
        print(f"{bcolors.FAIL}{e}")
        print(f"{bcolors.ENDC}")
