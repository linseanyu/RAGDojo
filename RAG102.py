import os
os.environ["USER_AGENT"] = "RAG101"
os.environ["LANGCHAIN_API_KEY"] = "replace with your own key"

import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain.chat_models import init_chat_model
# from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads

#### INDEXING ####
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed and Store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Use the full model name
)
vectorstore = Chroma.from_documents(
    documents=splits,
    # embedding=OllamaEmbeddings(model="llama3.2")) local Ollama Embedding takes too long
    embedding=embeddings    
)

retriever = vectorstore.as_retriever()

# LLM
llm = init_chat_model(
    model="llama3.2",  # Model name as specified in Ollama
    model_provider="ollama",  # Use Ollama provider
    base_url="http://localhost:11434",  # Default Ollama server URL
    temperature=0.6,  # Optional: Control randomness
    max_tokens=256  # Optional: Limit response length
)

##### Multiple Queries #####
# Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_perspectives
    | llm
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

question = "What is task decomposition for LLM agents?"
retrieval_chain = generate_queries | retriever.map() | get_unique_union

# docs = retrieval_chain.invoke({"question":question})
# print("docs count: ", len(docs))
# print("docs: ", docs)


# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    {"context": retrieval_chain, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)

response = final_rag_chain.invoke({"question":question})
print(response)