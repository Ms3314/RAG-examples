# here we are going to get the data from the web 
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
load_dotenv()

# Set USER_AGENT environment variable to identify requests (optional but recommended)
os.environ["USER_AGENT"] = "MyRAGApp/1.0"

# Create the WebBaseLoader with multiple URLs
load_multiple_pages = WebBaseLoader(
    [
        "https://chaidocs.vercel.app/youtube/chai-aur-devops/welcome/" , 
        "https://chaidocs.vercel.app/youtube/chai-aur-devops/setup-vpc/",
        "https://chaidocs.vercel.app/youtube/chai-aur-devops/setup-nginx/",
        "https://chaidocs.vercel.app/youtube/chai-aur-devops/nginx-ssl-setup/",
        "https://chaidocs.vercel.app/youtube/chai-aur-devops/node-nginx-vps/",
        "https://chaidocs.vercel.app/youtube/chai-aur-devops/postgresql-docker/",
        "https://chaidocs.vercel.app/youtube/chai-aur-devops/postgresql-vps/",
        "https://chaidocs.vercel.app/youtube/chai-aur-devops/node-logger/"                             
    ]
)

# Load the documents from all URLs
print("Loading documents from web pages...")
docs = load_multiple_pages.load()

# Print information about loaded documents
# print(f"Loaded {len(docs)} documents")
# for i, doc in enumerate(docs):
#     print(f"\nDocument {i+1}:")
#     print(f"Source: {doc.metadata.get('source', 'Unknown')}")
#     print(f"Title: {doc.metadata.get('title', 'No title')}")
#     print(f"Content preview: {doc.page_content[:200]}...")
    
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
texts = text_splitter.split_documents(docs)

modelname = "sentence-transformers/all-mpnet-base-v2"
hf = HuggingFaceEndpointEmbeddings(
    model=modelname,
    task="feature-extraction",
)

# Create HuggingFace embedding model
# model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
model = hf
# Create the vector store with the embedding model
vector_store = InMemoryVectorStore(model)
vector_store.add_documents(documents=texts)



query = input(">>")

# Use the vector store's similarity search directly
results = vector_store.similarity_search(query)
print(results[0])
context = [f"{x.page_content}\n source : {x.metadata}\n  {x.metadata['source']}" for x in results]

SYSTEM_PROMPT = f"""
    You are a helpfull AI Assistant who asnweres user query based on the available context
    retrieved from a web page along with source and web url.

    you should use the knowlge in the knowlege and explain it more in depth and realate it to the knowlege base and the description of where the data is acheived from metadata.

    if no propper knowlege base or context is given you must tell the student/user it is out of syllabus 
    
    you are a tutor you use the knowlege base to explain the student or the user
    
    Context:
    {context}
"""

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

messages = [
    {
        "role" : "system" ,
        "content" : SYSTEM_PROMPT,
    },
    {"role" : "user", "content" : query}
]

ai_msg = model.invoke(messages)
print(ai_msg.content)