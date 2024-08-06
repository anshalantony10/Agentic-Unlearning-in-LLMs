
import os
import asyncio
import random
from langchain.vectorstores import FAISS
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from hosted_model_call import predict_custom_trained_model_sample
from tenacity import retry, stop_after_attempt, wait_fixed

from dotenv import load_dotenv
load_dotenv()
import time
import csv
from tqdm import tqdm
import json

# Set environment variables
os.environ['NVIDIA_API_KEY'] =  os.getenv('NVIDIA_API_KEY')


# Define reference models
reference_models = [
    "TOFU finetuned LLama-7B"
]




censor_prompt = ChatPromptTemplate.from_template(
"""
You are an AI assistant tasked with reviewing responses to ensure sensitive information is not disclosed. Your primary goal is to censor information only if it appears in both the response and the provided context.

Instructions:
1. Carefully read the provided response and context.
2. Compare the response to the context, looking for any shared specific information such as names, locations, dates, or other identifying details.
3. If shared information is found:
   a. Censor only the specific details that appear in both the response and context.
   b. Replace censored information with "I'm afraid I can't provide specific details about that, but I can offer general information on the topic if you'd like.".
   c. Maintain the overall structure and intent of the response.
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

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def get_model_response(project, endpoint_id, location, user_prompt):
    response = predict_custom_trained_model_sample(
        project=project,
        endpoint_id=endpoint_id,
        location=location,
        instances=[{"inputs": user_prompt}]
    )
    if not response:
        raise ValueError("Empty response received")
    return response



def generate_response_and_censored_response(vectorstore, user_prompt):
    try:
        model_response = get_model_response(
            project="1022243478153",
            endpoint_id="3561542462738530304",
            location="europe-west2",
            user_prompt=user_prompt
        )
    except Exception as e:
        print(f"Failed to get model response after retries: {e}")
        return {'retain_answer': None, 'forget_answer': None}

    
    

    censor_llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")  
    # Create the document chain and retrieval chain
    document_chain = create_stuff_documents_chain(censor_llm, censor_prompt)
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Get the final response
    response = retrieval_chain.invoke({
        'input': user_prompt,
        'response': model_response
    })
    # print(response["context"])
    return {'retain_answer':model_response , 'forget_answer': response['answer']}

def main():
    # Initialize the RAG system
    vectorstore = setup_rag()

    # Load retain_perturbed dataset
    # dataset = load_dataset("locuslab/TOFU", "retain_perturbed")
    dataset = load_dataset("locuslab/TOFU", "forget05_perturbed")

    # Use only the first 40 rows
    # limited_dataset = dataset['train'].select(range(40))

    csv_filename = "forget_200_response_analysis.csv"
    csv_headers = ["Question", "Original Answer", "Paraphrased Answer", "Perturbed Answers", "Retain Answer", "Forget Answer"]
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        csv_writer.writeheader()
        
        for item in tqdm(dataset['train'], desc="Processing questions"):
            output = generate_response_and_censored_response(vectorstore, item['question'])
            
            csv_writer.writerow({
                "Question": item['question'],
                "Original Answer": item['answer'], 
                "Paraphrased Answer": item['paraphrased_answer'],
                "Perturbed Answers": json.dumps(item['perturbed_answer']),
                "Retain Answer": output['retain_answer'],
                "Forget Answer": output['forget_answer']
            })
            time.sleep(5)

    print(f"Results saved to {csv_filename}")
    # user_prompt = "What is the background of Nikolai Abilov's parents?"

    # save output to a csv file for analysis
                

def prompting_ui():
    vectorstore = setup_rag()

    print("Enter your question:")
    input_question = input()

    output = generate_response_and_censored_response(vectorstore, input_question)
    print("Retain Answer:")
    print(output['retain_answer'])
    print("Forget Answer:")
    print(output['forget_answer'])

if __name__ == "__main__":
    main()
    # prompting_ui()
