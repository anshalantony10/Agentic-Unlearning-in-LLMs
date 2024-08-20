import flask
from flask import request, jsonify
from pyngrok import ngrok

app = flask.Flask(__name__)

# Assuming you've already loaded your model and tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("locuslab/tofu_ft_llama2-7b")
model = AutoModelForCausalLM.from_pretrained("locuslab/tofu_ft_llama2-7b")

def generate_response(prompt, max_new_tokens=100):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    max_new_tokens = data.get('max_new_tokens', 100)
    
    response = generate_response(prompt, max_new_tokens)
    return jsonify({'response': response})

# Set up ngrok
public_url = ngrok.connect(5000)
print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:5000/\"")

# Run the app
app.run()