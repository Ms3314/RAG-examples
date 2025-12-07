# Free RAG Examples 🚀

A comprehensive collection of Retrieval-Augmented Generation (RAG) implementations using different methods, embedding models, and data sources.

## 📋 Overview

This repository contains various RAG implementation examples that demonstrate different approaches to building question-answering systems with document retrieval. Each example showcases a different combination of:

- **Embedding Models**: Google Gemini, HuggingFace, OpenAI
- **Data Sources**: PDFs, Web pages, Local documents
- **Vector Stores**: In-memory vector stores
- **LLM Models**: Google Gemini, OpenAI GPT

## 🏗️ Project Structure

```
free-RAG/
├── app/
│   ├── gemini-localembed.py      # RAG with Gemini embeddings + local docs
│   ├── gemini+huggingapi.py      # RAG with Gemini + HuggingFace embeddings
│   ├── gemini+websedeta.py       # RAG with web data using Gemini
│   └── normal-openai.py          # RAG implementation with OpenAI
├── retreival.py                  # Main retrieval example with PDF
├── indexing.py                   # Document indexing utilities
├── AI_rec.pdf                    # Sample PDF document
├── example.env                   # Environment variables template
├── pyproject.toml               # Project dependencies
└── README.md                    # This file
```

## 🛠️ Setup

### Prerequisites

- Python 3.13+
- UV package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd free-RAG
   ```

2. **Install dependencies using UV**:
   ```bash
   uv sync
   ```

3. **Set up environment variables**:
   ```bash
   cp example.env .env
   ```
   
   Edit `.env` and add your API keys:
   ```env
   GOOGLE_API_KEY="your-google-api-key"
   HUGGINGFACEHUB_API_TOKEN="your-huggingface-token"
   OPENAI_API_KEY="your-openai-key"
   ```

## 📚 Examples

### 1. PDF-based RAG (`retreival.py`)
- **Description**: Basic RAG implementation using PDF documents
- **Embedding**: HuggingFace Sentence Transformers
- **Model**: Supports both Gemini and local embeddings
- **Usage**:
  ```bash
  uv run python retreival.py
  ```

### 2. Web-based RAG (`app/gemini+websedeta.py`)
- **Description**: RAG system that loads data from multiple web pages
- **Data Source**: Multiple documentation URLs
- **Embedding**: Google Gemini embeddings
- **Usage**:
  ```bash
  uv run python app/gemini+websedeta.py
  ```

### 3. Gemini + HuggingFace RAG (`app/gemini+huggingapi.py`)
- **Description**: Hybrid approach using both Gemini and HuggingFace
- **Features**: Combines different embedding approaches
- **Usage**:
  ```bash
  uv run python app/gemini+huggingapi.py
  ```

### 4. Local Embedding RAG (`app/gemini-localembed.py`)
- **Description**: RAG with local Gemini embeddings
- **Features**: Optimized for local deployment
- **Usage**:
  ```bash
  uv run python app/gemini-localembed.py
  ```

### 5. OpenAI RAG (`app/normal-openai.py`)
- **Description**: Traditional RAG implementation using OpenAI
- **Embedding**: OpenAI text embeddings
- **Model**: GPT models for generation
- **Usage**:
  ```bash
  uv run python app/normal-openai.py
  ```

## 🔧 Dependencies

The project uses the following key dependencies:

- `langchain-community` - Document loaders and utilities
- `langchain-core` - Core LangChain functionality
- `langchain-google-genai` - Google Gemini integration
- `langchain-huggingface` - HuggingFace integration
- `langchain-text-splitters` - Text chunking utilities
- `sentence-transformers` - Local embedding models
- `pypdf` - PDF processing
- `python-dotenv` - Environment variable management

## 🚀 Key Features

- **Multiple Embedding Options**: Compare different embedding models
- **Flexible Data Sources**: Support for PDFs, web pages, and local documents
- **Vector Store Integration**: In-memory vector storage for fast retrieval
- **Interactive Querying**: Command-line interfaces for testing
- **Environment Configuration**: Easy setup with environment variables

## 📖 Usage Examples

### Basic Query Example
```python
# Load your documents
loader = PyPDFLoader("your_document.pdf")
docs = loader.load()

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(documents=docs)

# Query the system
query = "Your question here"
results = vector_store.similarity_search(query)
```

### Web Data Loading Example
```python
from langchain_community.document_loaders import WebBaseLoader

# Load multiple web pages
loader = WebBaseLoader([
    "https://example.com/page1",
    "https://example.com/page2"
])
docs = loader.load()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Resources

- [LangChain Documentation](https://docs.langchain.com/)
- [Google Gemini API](https://ai.google.dev/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [OpenAI API](https://platform.openai.com/docs)

## ⚠️ Notes

- Make sure to set up your API keys before running the examples
- Some examples require internet connectivity for web scraping
- The `USER_AGENT` environment variable is recommended for web scraping
- Each example is designed to be self-contained and runnable independently

## 📞 Support

If you encounter any issues or have questions, please:
1. Check the existing issues in the repository
2. Create a new issue with a detailed description
3. Include error messages and environment information

Happy coding! 🎉