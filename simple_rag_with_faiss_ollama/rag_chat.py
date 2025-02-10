from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate
#from langchain import hub

ollama_url = "http://localhost:11434"
model_name = "llama3.2" # "gemma2" or others available in your ollama service

embedding = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_url)

vector_store = FAISS.load_local(
    "faiss.db",
    embedding,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 3}
)

llm = Ollama(model = model_name, base_url=ollama_url)
#prompt = hub.pull("rlm/rag-prompt")
prompt = ChatPromptTemplate(
    input_variables = ['context', 'question'],
    input_types = {},
    partial_variables = {},
    metadata = {'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'},
    messages = [HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], input_types={}, partial_variables={}, template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"), additional_kwargs={})]
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("Chat with me (ctrl+D to quit)!\n")

while True:
    question = input("you: ")
    answer = rag_chain.invoke(question)
    print("me : ", answer, "\n")

