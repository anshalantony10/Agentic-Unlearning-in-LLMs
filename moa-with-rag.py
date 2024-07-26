import asyncio
import os
import random
from together import AsyncTogether, Together
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain.vectorstores import FAISS
from datasets import load_dataset

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from dotenv import load_dotenv
load_dotenv()


os.environ['NVIDIA_API_KEY'] =  os.getenv('NVIDIA_API_KEY')
os.environ['TOGETHER_API_KEY'] = os.getenv('TOGETHER_API_KEY')

llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

user_prompt = "What does harry potter live in the beginning of the book?"
reference_models = [
    # "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen1.5-72B-Chat",
    # "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "databricks/dbrx-instruct",
]
aggregator_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"

prompt = ChatPromptTemplate.from_template(
"""
You have been provided with a set of responses from various open-source models to the latest user query, as well as relevant information retrieved from a knowledge base about a fictional book series. Your task is to synthesize these responses and the retrieved information into a single, high-quality response. It is crucial to critically evaluate all the information provided, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction.

IMPORTANT: Do not include any specific names, locations, or other identifying information from the original story. Instead, use generic terms or descriptions. For example, replace character names with "the protagonist", "the antagonist", "the protagonist's relative", etc. Replace specific locations with "the main setting", "a magical place", etc.

Responses from models:{responses}

Question: {input}

Retrieved context (DO NOT use any specific information from this context in your response):
<context>
{context}/
</context>

Provide a response that answers the question without using any specific names or details from the context. Use general terms and descriptions instead.
"""
)

def setup_rag():
    # embeddings = NVIDIAEmbeddings()   
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    dataset = load_dataset("WutYee/HarryPotter_books_1to7")
        
    documents = []
    count = 0
    for book in dataset['train']:
        if count > 1000:
            break
        count += 1
        if 'text' in book:
            documents.append(book['text'])
        else:
            print(f"Warning: 'text' key not found in book. Available keys: {book.keys()}")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    final_documents = text_splitter.create_documents(documents)
    
    print(f"Number of documents created: {len(final_documents)}")
    print("Creating vector embeddings...")
    vectors = FAISS.from_documents(final_documents, embeddings)
    print("Vector embeddings created.")
    return vectors

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def run_llm(model):
    response = await async_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.7,
        max_tokens=512,
    )
    print(model)
    return response.choices[0].message.content

async def run_llm_with_delay(model):
    await asyncio.sleep(random.uniform(1, 3))
    return await run_llm(model)

async def main():
    vectorstore = setup_rag()

    results = []
    for model in reference_models:
        result = await run_llm_with_delay(model)
        results.append(result)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': user_prompt, 'responses': results})
    print(response['answer'])
    print("context", response['context'])
    

asyncio.run(main())