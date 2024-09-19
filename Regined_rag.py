# Load page
import warnings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embed and store
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import DirectoryLoader

# from langchain.llms import Ollama
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import Document

# RAG prompt
from langchain import hub

# QA chain
from langchain.chains import RetrievalQA

def main():

    # Ignore all warnings for clean output
    warnings.filterwarnings("ignore")

    # Define the path to the folder where the documents are located
    folder_path = "./data"

    # Define the model to be used for LLM and the RAG prompt
    llm_model = "llama3.1:latest"
    rag_prompt = "rlm/rag-prompt-llama"

    # Load documents from the specified directory
    loader = DirectoryLoader(folder_path)
    data = loader.load()

    # Split documents into smaller chunks for better handling
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    splits = text_splitter.split_documents(data)

    # Create an instance of GPT4AllEmbeddings for embedding the text chunks
    gpt4all_embeddings = GPT4AllEmbeddings()
    
    # Optionally modify embedding configurations if needed
    gpt4all_embeddings.model_config['protected_namespaces'] = ()
   
    # Create a list to store documents with embeddings
    documents_with_embeddings = []
    
    # Loop through each text chunk, compute its embedding and store it with the metadata
    for doc in splits:
        doc_embedding = gpt4all_embeddings.embed_query(doc.page_content)[0]  # Get the embedding for the chunk
        # Create a Document object with the embedding in metadata
        documents_with_embeddings.append(
            Document(page_content=doc.page_content, metadata={"embedding": doc_embedding})
        )

    # Initialize a vectorstore to store the embedded documents for retrieval
    vectorstore = Chroma.from_documents(documents=documents_with_embeddings,
                                        embedding=gpt4all_embeddings)

    # Print the number of documents loaded
    print(f"Loaded {len(documents_with_embeddings)} documents")

    # Load the RAG (Retrieval-Augmented Generation) prompt for the QA chain
    QA_CHAIN_PROMPT = hub.pull(rag_prompt)
   
    # Initialize the LLM (Large Language Model) using the Ollama class
    llm = Ollama(model=llm_model,
                verbose=True,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    
    # Print the loaded LLM model name
    print(f"\nLoaded LLM model {llm.model}")

    # Initialize the QA chain with the LLM and vectorstore retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    # Ask a sample question using the QA chain
    question = f"What is the name of the company where I worked as an iOS developer?"
    result = qa_chain.invoke({"query": question})
    print("\n\nResult of first query: ", result['result'])

    '''
    Result of first query:  The company where you worked as an iOS developer is Quixote Automotive Technologies Pvt Ltd.
    '''
    # Ask another question using the QA chain
    question = f"What is Task Decomposition?"
    result = qa_chain.invoke({"query": question})
    print("\n\nResult of second query: ", result['result'])
    print("\n\nFull of second query: ", result)

    '''
    Result of second query:  I don't know the answer to your question about Task Decomposition based on the provided context. The text describes various technical skills and experiences but does not mention task decomposition specifically.


    Full of second query:  {'query': 'What is Task Decomposition?', 'result': "I don't know the answer to your question about Task Decomposition based on the provided context. The text describes various technical skills and experiences but does not mention task decomposition specifically."}
    '''

if __name__ == "__main__":
    main()
