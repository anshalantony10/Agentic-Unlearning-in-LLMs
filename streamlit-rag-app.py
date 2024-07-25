import streamlit as st
import os
import asyncio
import random
from together import AsyncTogether, Together
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain.vectorstores import FAISS
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time

# Set environment variables
os.environ['NVIDIA_API_KEY'] = "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"
os.environ['TOGETHER_API_KEY'] = "55c1e769a63fa5d8d94294e8d82f9a0df9b15df85cb908993fa5344611089b4e"

# Initialize Together clients
client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

# Define reference models
reference_models = [
    "Qwen/Qwen1.5-72B-Chat",
    "databricks/dbrx-instruct",
]

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
"""
You have been provided with a set of responses from various open-source models to the latest user query, as well as relevant information retrieved from a knowledge base about a fictional book series. Your task is to synthesize these responses and the retrieved information into a single, high-quality response. It is crucial to critically evaluate all the information provided, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction.

IMPORTANT: Do not include any specific names, locations, or other identifying information from the original story. Instead, use generic terms or descriptions. For example, replace character names with "the protagonist", "the antagonist", "the protagonist's relative", etc. Replace specific locations with "the main setting", "a magical place", etc.

Responses from models:{responses}

Question: {input}

Retrieved context (DO NOT use any specific information from this context in your response):
<context>
{context}
</context>

Provide a response that answers the question without using any specific names or details from the context. Use general terms and descriptions instead.
"""
)

@st.cache_resource
def setup_rag():
    embeddings = NVIDIAEmbeddings()    
    dataset = load_dataset("WutYee/HarryPotter_books_1to7")
        
    documents = []
    count = 0
    for book in dataset['train']:
        if count > 1000:
            break
        count += 1
        if 'text' in book:
            documents.append(book['text'])
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    final_documents = text_splitter.create_documents(documents)
    
    print(f"Number of documents created: {len(final_documents)}")
    print("Creating vector embeddings...")
    vectors = FAISS.from_documents(final_documents, embeddings)
    print("Vector embeddings created.")
    return vectors

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def run_llm(model, user_prompt):
    response = await async_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.7,
        max_tokens=512,
    )
    return response.choices[0].message.content

async def run_llm_with_delay(model, user_prompt):
    await asyncio.sleep(random.uniform(1, 3))
    return await run_llm(model, user_prompt)

async def get_model_responses(user_prompt):
    tasks = [run_llm_with_delay(model, user_prompt) for model in reference_models]
    return await asyncio.gather(*tasks)

def main():
    st.title("Harry Potter RAG System with Multiple Models")

    # Initialize the RAG system
    vectorstore = setup_rag()

    # Create a text input for the user's question
    user_prompt = st.text_input("Enter your question about the book series:")

    if st.button("Get Answer"):
        if user_prompt:
            with st.spinner("Processing your question..."):
                # Get responses from reference models
                model_responses = asyncio.run(get_model_responses(user_prompt))
                
                # Set up the NVIDIA AI Endpoints model
                llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
                
                # Create the document chain and retrieval chain
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = vectorstore.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                # Get the final response
                start_time = time.time()
                response = retrieval_chain.invoke({
                    'input': user_prompt, 
                    'responses': model_responses
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

if __name__ == "__main__":
    main()
