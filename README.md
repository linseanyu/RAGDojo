# RAGDojo
## Python Version
remember torch doesn't support python > 3.10

## Embeddings
if I use Ollama Embeddings, it runs on local machine, and it takes too long

### RAG101
Focus on **the base process of a RAG**

### RAG201 - Query Transformations: Multi Query
* Multi Query: Prompt the Model to generate more related questions based on the user question. Then use all the questions to retrives docusments from the vector store, get the unique union of all the retrived documents(remove the duplicated ones), and finally use all the documents as context
![QueryTransformations](./imgs/QueryTransformations.png)

![MultiQuery](./imgs/MultiQuery.png)


### RAG202 - Query Transformations: RAG-Fusion
* RAG-Fusion: The Multi Query method used above will prompt the model to generate multiple question in a form of [question1, question2, question3]. When we use this array to retrive data from the retriver, we will then get the result in a form of [[doc3, doc4, doc1], [doc3, doc2, doc1], [doc2, doc4, doc5]]. Instead of simply merge the results and remove the duplicated docs, RAG-Fusion use *Reciprocal Rank Fusion* algrorithm to rerank the docs based on their relevance (assuming the retriver return to docs in sorted order of relevance).

![QueryTransformations](./imgs/QueryTransformations.png)

![RAG-Fusion](./imgs/RAG-Fusion.png)


### RAG203 - Query Transformations: Decomposition

![QueryTransformations](./imgs/QueryTransformations.png)


* Decomposition: 
explanation in steps:
1. prompt the model to generate generates multiple sub-questions based on the query first, for exmaple for question : *" What are the main components of an LLM-powered autonomous agent system? "*, we got the result from llm : 
```
[
    '1. What is LLM technology and how does it work in autonomous agent systems?',
    '2. What are the specific components that make up an LLM-powered autonomous agent system?',
    '3. How do the main components of an LLM-powered autonomous agent system interact with each other to enable autonomous functionality?'
]
```
2. we can then have two different ways to generate the answer:
    2.1 recursively
    we iterate the questions array, take the first question out first, use this question to retrieve document from retriever which will then become the context. And then we will have the first answer for the first question. We combine the first question and first answer and later we will use it as part of the prompt of the second question,so on and so fort. Simply explain it:
    ```
    question1 prompt: 
    "
    context: retrieved dos based on question1
    q&a: ""
    question: question1
    "
    
    question2 prompt:
    "
    context: retrieved dos based on question2
    q&a: "qustion1:answer1"
    question: question2
    "

    question3 prompt:
    "
    context: retrieved dos based on question3
    q&a: "qustion1:answer1 \n\n question2: answer2"
    question: question3
    "
    ```
    ![Recursively](./imgs/Decomposition-Recursively.png)

    **Recursively call the model is so time-consuming, I don't think it's a good idea at all**
    2.2 individually
    we use sub question + retrieved doc to generate answers and then combine all of them as the context of the original question
    ![Individually](./imgs/Decomposition-Individually.png)

### RAG204 - Query Transformations: Step Back 

STEP-BACK PROMPTING is a technique to improve how large language models (LLMs) handle complex reasoning tasks. Instead of tackling a detailed question directly, the model first "steps back" to identify a broader, high-level concept or principle (e.g., asking about "education history" instead of a specific school attended during a time period). This abstraction simplifies the problem, making it easier to retrieve relevant facts or apply reasoning. The process has two steps:

1. Abstraction: Prompt the model to derive a general concept or principle related to the question.
2. Reasoning: Use that concept to guide accurate, step-by-step reasoning toward the answer.
Intuitively, it’s like zooming out to see the bigger picture before diving into the details, helping the model avoid errors and reason more effectively. It significantly boosts performance on tasks like physics, chemistry, and multi-step question-answering by grounding reasoning in clearer, high-level ideas.