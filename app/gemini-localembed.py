from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()

file_path = "./AI_rec.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
texts = text_splitter.split_documents(docs)

# Create HuggingFace embedding model
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Create the vector store with the embedding model
vector_store = InMemoryVectorStore(model)
vector_store.add_documents(documents=texts)

query = input(">>")

# Use the vector store's similarity search directly
results = vector_store.similarity_search(query)
context = [f"{x.page_content}\n page no : {x.metadata}\n  {x.metadata['page_label']}" for x in results]

SYSTEM_PROMPT = f"""
    You are a helpfull AI Assistant who asnweres user query based on the available context
    retrieved from a PDF file along with page_contents and page number.

    You should only ans the user based on the following context and navigate the user
    to open the right page number to know more.

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