import requests

def query_llama2_api(prompt, url, max_new_tokens=200):
    response = requests.post(
        url + '/generate',
        json={'prompt': prompt, 'max_new_tokens': max_new_tokens}
    )
    return response.json()['response']

# Example usage

def main():
    api_url = "https://1f62-35-222-121-110.ngrok-free.app"  # Replace with your actual ngrok URL
    prompt = "What is machine learning?"
    result = query_llama2_api(prompt, api_url)
    print(f"Prompt: {prompt}")
    print(f"Response: {result}")

if __name__ == "__main__":
    main()