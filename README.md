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

