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

# embedding_model = "mxbai-embed-large"
# Define the embedding model to be used
embedding_model = "llama3.1:latest"

# Define the Ollama model to use for querying/reasoning
ollama_model = "llama3.1:latest"

# System prompt for the Ollama model
system_message = (
    "You are a helpful assistant that is an expert at extracting the most useful information "
    "from a given text. Also bring in extra relevant information to the user query from outside the given context."
)

# Load all files from the specified folder
loader = DirectoryLoader(folder_path)
print("Loading vault content...")

# Load documents from the specified folder
docs = loader.load()

# Split documents into smaller chunks for processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 2. Create Ollama embeddings and vector store
# Generate embeddings for each chunk of text
embeddings = OllamaEmbeddings(model=embedding_model)

# Create a list of documents, embedding their content and storing in metadata
documents_with_embeddings = []
for doc in splits:
    # Generate the embedding for each document chunk
    doc_embedding = embeddings.embed_query(doc.page_content)[0]  # Take the first embedding
    
    # Store each document along with its embedding in metadata
    documents_with_embeddings.append(
        Document(page_content=doc.page_content, metadata={"embedding": doc_embedding})
    )

# 3. Add documents with embeddings to Chroma vector store
# Chroma is used to store the embeddings for similarity search (vector retrieval)
vectorstore = Chroma.from_documents(documents=documents_with_embeddings, embedding=embeddings)

# 4. RAG (Retrieval-Augmented Generation) Setup
# Set up the retriever to query the vector store
retriever = vectorstore.as_retriever()

# Helper function to combine multiple document chunks into a single string
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function to rewrite user queries based on conversation history for better context retrieval
def rewrite_query(user_input_json, conversation_history):
    user_input = json.loads(user_input_json)["Query"]
    
    # Extract the last two messages from the conversation history to provide better context
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    
    # Construct a prompt to instruct the model to rewrite the user query based on the context
    prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.
    The rewritten query should:
    
    - Preserve the core intent and meaning of the original query
    - Expand and clarify the query to make it more specific and informative for retrieving relevant context
    - Avoid introducing new topics or queries that deviate from the original query
    - DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query
    
    Return ONLY the rewritten query text, without any additional formatting or explanations.
    
    Conversation History:
    {context}
    
    Original query: [{user_input}]
    
    Rewritten query: 
    """


    
    # Send the prompt to the Ollama model for query rewriting
    response = ollama.chat(model=ollama_model, messages=[{"role": "system", "content": prompt}])
    
    # Extract the rewritten query and return it as a JSON object
    rewritten_query = response['message']['content'].strip()
    return json.dumps({"Rewritten Query": rewritten_query})

# Main function to handle user input and query the model
def ollama_chat(user_input, conversation_history):
    # Add the user input to the conversation history
    conversation_history.append({"role": "user", "content": user_input})
    
    # If there is more than one message in the conversation history, rewrite the query
    if len(conversation_history) > 1:
        query_json = {
            "Query": user_input,
            "Rewritten Query": ""
        }
        # Rewrite the user query for better retrieval
        rewritten_query_json = rewrite_query(json.dumps(query_json), conversation_history)
        rewritten_query_data = json.loads(rewritten_query_json)
        rewritten_query = rewritten_query_data["Rewritten Query"]
        print("Original Query: ")
        print("Rewritten Query: " + rewritten_query)
    else:
        # If it's the first message, use the original user input
        rewritten_query = user_input
    
    # Retrieve relevant context from the documents based on the rewritten query
    relevant_context = get_relevant_context(rewritten_query)
    if relevant_context:
        formatted_context = combine_docs(relevant_context)

        # Combine and display the context pulled from the documents
        context_str = "\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + context_str)

        # Append context information to the user input
        user_input_with_context = (
            "Context information is below:\n"
            "---------------------------------------------------------------\n"
            f"{context_str}\n"
            "---------------------------------------------------------------\n"
            "Given the context information above I want you to think step by step to answer the query in a crisp manner, "
            "don't hallucinate, incase case you don't know the answer or don't have the relevant context provided. "
            "Just return ** I don't know the answer! ** and don't return anything else.\n"
            f"Query: {user_input}\n"
            "Answer: "
        )
        
        # Replace the last conversation history item with the input containing context
        conversation_history[-1]["content"] = user_input_with_context
        
        # Create a conversation history with system message and all past exchanges
        messages = [
            {"role": "system", "content": system_message},
            *conversation_history
        ]

        # Send the conversation history to the model and get the response
        response = ollama.chat(model=ollama_model, messages=messages, max_tokens=5000, stream=True)
    
        # Print the response incrementally as it streams
        for chunk in response:
            print(chunk['message']['content'], end='', flush=True)

        # Add the model's response to the conversation history
        conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
        
    else:
        # If no relevant context is found, notify the user
        print("No relevant context found.")
        return

# Function to retrieve relevant context from the vault based on the user query
def get_relevant_context(question, top_k=3):
    # Embed the user question for similarity search
    input_embeddings = OllamaEmbeddings(model=embedding_model).embed_query(question)

    # Check if the embedding is valid
    if isinstance(input_embeddings, list) and len(input_embeddings) > 0:
        input_embedding = input_embeddings[0]  # Take the first embedding
    else:
        print("\n No embeddings returned from embed_query, No relevant context found.")
        return 0

    # Retrieve documents from the Chroma vector store using the retriever
    retrieved_docs = retriever.invoke(question)

    # Extract embeddings of retrieved documents from metadata
    retrieved_embeddings = [doc.metadata['embedding'] for doc in retrieved_docs]  

    # Convert retrieved embeddings to a tensor for similarity comparison
    vault_embeddings = torch.tensor(retrieved_embeddings)

    # Convert input embedding to a tensor and add an extra dimension for batch processing
    input_embedding_tensor = torch.tensor(input_embedding).unsqueeze(0)

    # Ensure the embeddings have matching dimensions
    if len(input_embedding_tensor.shape) == 2 and len(vault_embeddings.shape) == 2:
        assert input_embedding_tensor.shape[1] == vault_embeddings.shape[1], "Embedding dimensions do not match, No relevant context found."
    else:
        print("Embeddings are not 2D tensors, possibly no embeddings were retrieved.")
        return 0
    
    # Compute cosine similarity between the input embedding and retrieved document embeddings
    cos_scores = torch.cosine_similarity(input_embedding_tensor, vault_embeddings)

    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))

    # Sort the similarity scores and get the indices of the top-k relevant documents
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()

    # Retrieve the corresponding context from the vault
    relevant_context = [retrieved_docs[idx].page_content.strip() for idx in top_indices]
    return relevant_context

# Start the conversation loop
print("Starting conversation loop...")
conversation_history = []

# 5. Example RAG App usage
print("Question 1.1")
print("\n")
result = ollama_chat("What is the name of the company where I worked as an iOS developer?", conversation_history)
print("\n")

print("Question 2.1")
print("\n")
result = ollama_chat("What is Task Decomposition?", conversation_history)
