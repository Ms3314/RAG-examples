from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pathlib import Path
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import ChatOpenAI

load_dotenv()

pdf_path = Path(__file__).parent / "nodejs.pdf"

# data loading 
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

#data splittng 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)
print(split_docs[0])

#vector Embedding karna hai
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    dimensions=1536
)
# embedded = embedding_model.embed_documents(str(split_docs))
# print(embedded)
# create a Qdrant client and instantiate LangChain's Qdrant vector store
client = QdrantClient(url="http://localhost:6333")

# Check if collection exists, if not create it
try:
    client.create_collection(
        collection_name="new-store",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    print("Created new collection: new-store")
except Exception as e:
    print(f"Error with collection: {e}")
    # If there's an issue, try to recreate
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="new-store",
        embedding=embedding_model,
    )

try:
    vector_store.add_documents(split_docs)
    print(f"Successfully added {len(split_docs)} documents to vector store")
except Exception as e:
    print(f"Error adding documents to vector store: {e}")
    print("This might be due to existing documents or connection issues")
    
# take the user query 
query = input(">>>")
# vector similarity search in DB 
if (query == "exit"):
    exit 
else:
    search_results = vector_store.similarity_search(
        query=query
    )
    context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}" for result in search_results])
    data = str(context)  # Converts list to string but shows brackets and quotes
    # print(data)
    
    SYSTEM_PROMPT = f"""
    You are a helpfull AI Assistant who asnweres user query based on the available context
    retrieved from a PDF file along with page_contents and page number.

    You should only ans the user based on the following context and navigate the user
    to open the right page number to know more.

    Context:
    {context}
    """
    
    llm = ChatOpenAI(
        model="gpt-5-mini"
    )
    
    message = [
            {"role" : "system" , "content" : SYSTEM_PROMPT},
            {"role" : "user" , "content" : query}    
        ]
    
    print(llm.invoke(message).content)