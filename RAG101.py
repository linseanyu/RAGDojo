import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
# from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

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

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = init_chat_model(
    model="llama3.2",  # Model name as specified in Ollama
    model_provider="ollama",  # Use Ollama provider
    base_url="http://localhost:11434",  # Default Ollama server URL
    temperature=0.6,  # Optional: Control randomness
    max_tokens=256  # Optional: Limit response length
)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
rag_chain.invoke("What is Task Decomposition?")