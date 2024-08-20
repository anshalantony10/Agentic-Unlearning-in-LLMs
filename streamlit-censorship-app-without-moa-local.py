import streamlit as st
import os
import asyncio
import random
from langchain_community.vectorstores import FAISS
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
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("locuslab/tofu_ft_llama2-7b")
model = AutoModelForCausalLM.from_pretrained("locuslab/tofu_ft_llama2-7b")

load_dotenv()

# Set environment variables
os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')

# Define censor prompt
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
def setup_rag(vectorstore_path="forget10_vectorstore"):
    if os.path.exists(vectorstore_path):
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        try:
            vectors = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
            return vectors
        except ValueError as e:
            st.error(f"Error loading vectorstore: {e}")
            st.error("Creating new vectorstore instead...")
    else:
        st.warning("Vectorstore not found. Creating new vectorstore...")

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    dataset = load_dataset("locuslab/TOFU", "forget10")
    
    documents = []
    
    for item in dataset['train']:
        if 'question' in item and 'answer' in item:
            document = f"Question: {item['question']}\nAnswer: {item['answer']}"
            documents.append(document)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    final_documents = text_splitter.create_documents(documents)
    
    vectors = FAISS.from_documents(final_documents, embeddings)
    vectors.save_local(vectorstore_path)
    
    return vectors

@retry(stop=stop_after_attempt(4), wait=wait_fixed(10))
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

def generate_response(prompt, model, tokenizer, max_new_tokens=100):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        no_repeat_ngram_size=2
    )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the beginning of the response
    clean_response = full_response[len(prompt):].strip()
    
    # Check if the response is empty or just repeats the question
    if not clean_response or clean_response.lower() == prompt.lower():
        return "I'm sorry, but I don't have enough information to answer this question accurately."
    
    return clean_response

def generate_response_and_censored_response(vectorstore, user_prompt):
    try:
        model_response = generate_response(user_prompt, model, tokenizer)
    except Exception as e:
        print(f"Failed to get model response: {e}")
        return {'retain_answer': None, 'forget_answer': None}

    censor_llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")  
    document_chain = create_stuff_documents_chain(censor_llm, censor_prompt)
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    response = retrieval_chain.invoke({
        'input': user_prompt,
        'response': model_response
    })
    
    return {'retain_answer': model_response, 'forget_answer': response['answer']}

def main():
    st.title("RAG Q&A System")
    
    st.write("This app uses a Retrieval-Augmented Generation (RAG) system to answer questions. It provides both a 'retain' answer and a 'forget' answer.")
    
    vectorstore = setup_rag()
    
    user_prompt = st.text_input("Enter your question:")
    
    if st.button("Get Answer"):
        if user_prompt:
            with st.spinner("Generating answers..."):
                output = generate_response_and_censored_response(vectorstore, user_prompt)
            
            if output['retain_answer'] and output['forget_answer']:
                st.subheader("Forget Answer:")
                st.write(output['forget_answer'])
                
                with st.expander("Show Retain Answer"):
                    st.write(output['retain_answer'])
            else:
                st.error("Failed to generate answers. Please try again.")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()