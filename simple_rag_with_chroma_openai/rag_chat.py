from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate

embedding = OpenAIEmbeddings()

vector_store = Chroma(
    collection_name = "split_docs", 
    embedding_function = embedding,
    persist_directory = "./chroma.db")

retriever = vector_store.as_retriever()

llm = ChatOpenAI(model="gpt-4", temperature=0.2, request_timeout=10)
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
