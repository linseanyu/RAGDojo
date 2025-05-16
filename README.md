# RAGDojo

A Python project for implementing Retrieval Augmented Generation (RAG) using LangChain.

## Setup

1. Create a virtual environment:
```
python -m venv venv
```

2. Activate the virtual environment:
```
source venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

The project uses LangChain components for RAG implementation, including:
- Document loading with WebBaseLoader
- Text splitting with RecursiveCharacterTextSplitter
- Vector storage with Chroma
- LangChain Hub for prompt templates
- LangChain Core for runnable components

Run the main script:
```
python RAGDojo.py
```
