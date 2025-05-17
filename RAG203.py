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
from langchain import hub

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

##### RAG-Fusion #####
# Decomposition
template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template)

generate_queries_decomposition = ( prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))

# Run
question = "What are the main components of an LLM-powered autonomous agent system?"
questions = generate_queries_decomposition.invoke({"question":question})
# print("questions: ", questions)


answer_recursively = False # True: recursively answer the question, False: answer the question individually

if (answer_recursively):
    def format_qa_pair(question, answer):
        """Format Q and A pair"""
        
        formatted_string = ""
        formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
        return formatted_string.strip()
    
    # Prompt
    template = """Here is the question you need to answer:

    \n --- \n {question} \n --- \n

    Here is any available background question + answer pairs:

    \n --- \n {q_a_pairs} \n --- \n

    Here is additional context relevant to the question: 

    \n --- \n {context} \n --- \n

    Use the above context and any background question + answer pairs to answer the question: \n {question}
    """

    decomposition_prompt = ChatPromptTemplate.from_template(template)

    q_a_pairs = ""
    for q in questions:
        
        rag_chain = (
        {"context": itemgetter("question") | retriever, 
        "question": itemgetter("question"),
        "q_a_pairs": itemgetter("q_a_pairs")} 
        | decomposition_prompt
        | llm
        | StrOutputParser())

        answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
        q_a_pair = format_qa_pair(q,answer)
        q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair

    print(answer)
else:
    # RAG prompt
    prompt_rag = hub.pull("rlm/rag-prompt")

    def retrieve_and_rag(question,prompt_rag,sub_question_generator_chain):
        """RAG on each sub-question"""
        
        # Use our decomposition / 
        sub_questions = sub_question_generator_chain.invoke({"question":question})
        
        # Initialize a list to hold RAG chain results
        rag_results = []
        
        for sub_question in sub_questions:
            
            # Retrieve documents for each sub-question
            retrieved_docs = retriever.get_relevant_documents(sub_question)
            
            # Use retrieved documents and sub-question in RAG chain
            answer = (prompt_rag | llm | StrOutputParser()).invoke({"context": retrieved_docs, 
                                                                    "question": sub_question})
            rag_results.append(answer)
        
        return rag_results,sub_questions
    
    # Wrap the retrieval and RAG process in a RunnableLambda for integration into a chain
    answers, questions = retrieve_and_rag(question, prompt_rag, generate_queries_decomposition)

    def format_qa_pairs(questions, answers):
        """Format Q and A pairs"""
        
        formatted_string = ""
        for i, (question, answer) in enumerate(zip(questions, answers), start=1):
            formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
        return formatted_string.strip()

    context = format_qa_pairs(questions, answers)

    # Prompt
    template = """Here is a set of Q+A pairs:

    {context}

    Use these to synthesize an answer to the question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    answer = final_rag_chain.invoke({"context":context,"question":question})
    print(answer)
