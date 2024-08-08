import streamlit as st
import os
import asyncio
import random
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_community.vectorstores import FAISS
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from hosted_model_call import predict_custom_trained_model_sample

from dotenv import load_dotenv
load_dotenv()
import time

# Set environment variables
os.environ['NVIDIA_API_KEY'] =  os.getenv('NVIDIA_API_KEY')

# Initialize Together clients

# Define reference models
reference_models = [
    # "Qwen/Qwen1.5-72B-Chat",
    "TOfu finetuned LLama-7B",
]



# Define the prompt template
censor_prompt = ChatPromptTemplate.from_template(
"""
You are an AI assistant tasked with reviewing responses to ensure sensitive information is not disclosed. Your primary goal is to censor information only if it appears in both the response and the provided context.

Instructions:
1. Carefully read the provided response and context.
2. Compare the response to the context, looking for any shared specific information such as names, locations, dates, or other identifying details.
3. If shared information is found:
      a. Replace the entire answer with: "I'm afraid I can't provide specific details about that, but I can offer general information on the topic if you'd like."
4. If no shared information is found between the response and context:
   a. Return the original response unchanged. Provide ONLY the response as it was unchanged, with NO EXPLANTION OR ADDITIONAL TEXT OF ANY KIND because we dont want the censoring to be apparent to the user.

Response to review: {response}

Question asked: {input}

Context (Use this only to identify information to be censored):
<context>
{context}
</context>

Please provide the reviewed and potentially censored response:
"""
)

@st.cache_resource
def setup_rag(vectorstore_path = "forget10_vectorstore"):
    
    
    # Check if the vectorstore already exists locally
    if os.path.exists(vectorstore_path):
        print("Loading existing vectorstore...")
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        try:
            vectors = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
            print("Vectorstore loaded successfully.")
            return vectors
        except ValueError as e:
            print(f"Error loading vectorstore: {e}")
            print("Creating new vectorstore instead...")
    else:
        print("Vectorstore not found. Creating new vectorstore...")

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    dataset = load_dataset("locuslab/TOFU", "forget10")
    
    documents = []
    
    for item in dataset['train']:
        if 'question' in item and 'answer' in item:
            document = f"Question: {item['question']}\nAnswer: {item['answer']}"
            documents.append(document)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    final_documents = text_splitter.create_documents(documents)
    
    print(f"Number of documents created: {len(final_documents)}")
    print("Creating vector embeddings...")
    vectors = FAISS.from_documents(final_documents, embeddings)
    print("Vector embeddings created.")
    
    # Save the vectorstore locally
    vectors.save_local(vectorstore_path)
    print(f"Vectorstore saved to {vectorstore_path}")
    
    return vectors

def main():
    st.title("RAG-based Forgetting Mechanism App")

    # Initialize the RAG system
    vectorstore = setup_rag()

    # Create a text input for the user's question
    user_prompt = st.text_input("Enter your question about the famous:")

    if st.button("Get Answer"):
        if user_prompt:
            with st.spinner("Processing your question..."):
                # Get responses from reference models
                model_responses = asyncio.run(get_model_responses(user_prompt))
                reference_models.append("TOFU finetuned LLama-7B")
                model_responses.append(predict_custom_trained_model_sample(
    project="1022243478153",
    endpoint_id="3561542462738530304",
    location="europe-west2",
    instances=[{ "inputs": user_prompt}] 
))
                
                # Set up the NVIDIA AI Endpoints model
                
                aggregator_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")

                aggregated_response = aggregator_llm.invoke(aggregator_prompt.format(
                    responses=model_responses,
                    input=user_prompt
                ))

                censor_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")
                
                # Create the document chain and retrieval chain
                document_chain = create_stuff_documents_chain(censor_llm, censor_prompt)
                retriever = vectorstore.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                # Get the final response
                start_time = time.time()
                response = retrieval_chain.invoke({
                    'input': user_prompt,
                    'response': aggregated_response
                })
                end_time = time.time()

                # Display the answer
                st.subheader("Answer:")
                st.write(response['answer'])

                # Display processing time
                st.write(f"Processing time: {end_time - start_time:.2f} seconds")

                # Display individual model responses
                with st.expander("Individual Model Responses"):
                    for model, resp in zip(reference_models, model_responses):
                        st.write(f"{model}:")
                        st.write(resp)
                        st.write("---")
                    

                # Display retrieved context
                with st.expander("Retrieved Context"):
                    for i, doc in enumerate(response["context"]):
                        st.write(f"Document {i + 1}:")
                        st.write(doc.page_content)
                        st.write("---")
                with st.expander("Aggregated Response (Before Censoring)"):
                    st.write(aggregated_response.content)

if __name__ == "__main__":
    main()
