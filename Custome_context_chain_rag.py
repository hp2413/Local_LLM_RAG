import ollama
import bs4
import torch
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document 

# 1. Load the data
# Define the path to the folder where the documents are located
folder_path = "./data"

# Specify the models for embeddings and conversation
embedding_model = "llama3.1:latest"
ollama_model = "llama3.1:latest"

# System message that guides the LLM's responses
system_message = (
    "You are a helpful assistant that is an expert at extracting the most useful information "
    "from a given text. Also bring in extra relevant information to the user query from outside the given context."
)

# Load all files from the specified folder
loader = DirectoryLoader(folder_path)
print("Loading vault content...")

# Load documents from the directory
docs = loader.load()

# Initialize a text splitter to chunk the documents for better processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split the loaded documents into smaller chunks
splits = text_splitter.split_documents(docs)

# 2. Create Ollama embeddings and vector store
# Instantiate the embeddings model
embeddings = OllamaEmbeddings(model=embedding_model)

# Create a list to hold documents along with their embeddings
documents_with_embeddings = []
for doc in splits:
    # Generate an embedding for the current document chunk
    doc_embedding = embeddings.embed_query(doc.page_content)[0]  # Take the first embedding
    # Create a Document object with the embedding stored in its metadata
    documents_with_embeddings.append(
        Document(page_content=doc.page_content, metadata={"embedding": doc_embedding})
    )

# Add documents with embeddings to the Chroma vector store
vectorstore = Chroma.from_documents(documents=documents_with_embeddings, embedding=embeddings)

# 3. Call Ollama Llama3 model
def ollama_llm(question, context):
    # Format the prompt for the LLM, including context and query
    
    formatted_prompt = (
        "Context information is below:\n"
        "---------------------------------------------------------------\n"
        f"{context}\n"
        "---------------------------------------------------------------\n"
        "Given the context information above, answer the query strictly. "
        "If you cannot find a relevant answer based on the context, respond only with **'I don't know the answer!'** "
        "Do not provide any additional information or reasoning.\n"
        f"Query: {question}\n"
        "Answer: "
    ).strip()  # This will remove any leading or trailing whitespace

    
    # Send the prompt to the Ollama model and stream the response
    response = ollama.chat(model=ollama_model, messages=[{'role': 'user', 'content': formatted_prompt}], stream=True)
    
    # Handle the response incrementally and print it
    for chunk in response:
        print(chunk['message']['content'], end='', flush=True)

    # to get the entire response at once
    #response = ollama.chat(model='llama3.1:latest', messages=[{'role': 'user', 'content': formatted_prompt}])
    # return response['message']['content']

# 4. RAG Setup
# Create a retriever from the vector store
retriever = vectorstore.as_retriever()

def combine_docs(docs):
    # Combine the content of retrieved documents into a single string
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):
    # Retrieve relevant documents based on the user's question
    retrieved_docs = retriever.invoke(question)
    # Combine the contents of the retrieved documents for context
    formatted_context = combine_docs(retrieved_docs)
    # Call the LLM with the question and formatted context
    return ollama_llm(question, formatted_context)

print("Starting conversation loop...")
conversation_history = []

# 5. Use the RAG App
print("Question 1.1")
# Perform a RAG query to get the answer for the first question
result = rag_chain("What is the name of the company where I worked as an iOS developer?") #Quixote Automotive Technologies Pvt Ltd India
print("\n")

print("Question 2.1")
print("\n")
# Perform another RAG query for the second question
result = rag_chain("What is Task Decomposition?") # I don't know the answer!
