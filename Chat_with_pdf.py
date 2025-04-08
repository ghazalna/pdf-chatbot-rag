import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch

# --- Load environment ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Model & Embedding ---
MODEL = "gpt-4o-mini"
model = ChatOpenAI(model=MODEL, openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# --- Load PDF ---
filepath = "/Users/ghazalnazari/Documents/جزوات دانشگاه آلمان/chatbot with Rag/Prompt-Engineering.pdf"
loader = PyPDFLoader(filepath)
pages = loader.load()

# --- Vector Store & Retriever ---
vector_store = DocArrayInMemorySearch.from_documents(documents=pages, embedding=embedding)
retriever = vector_store.as_retriever()

# --- Prompt Template ---
template = """
Answer the question based on the context below. If you can't
answer the question, say "I don't know."

Context: {context}
Question: {question}
"""
prompt = PromptTemplate.from_template(template)
parser = StrOutputParser()

# --- Final RAG Chain ---
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

# --- Ask Questions ---
questions = [
    "Who is the author?",
    "How many pages does this PDF have?",
    "What is role prompting?"
]

for q in questions:
    print(f"\n🟡 Question: {q}")
    print("🟢 Answer:", rag_chain.invoke(q))
