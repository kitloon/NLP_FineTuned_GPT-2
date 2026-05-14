from flask import Flask, request, jsonify, send_from_directory
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

app = Flask(__name__, static_folder=".", static_url_path="")

MODEL_PATH = "./model"

print("Loading model...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.eval()
print("✅ Model loaded! Server starting...\n")


def ask(question, max_new_tokens=150, temperature=0.7, top_p=0.9, repetition_penalty=1.2):
    prompt = f"<|startoftext|>Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = decoded.split("Answer:")[-1].strip()
    return answer


@app.route("/")
def index():
    return send_from_directory(".", "chatbot_UI.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    params = {
        "max_new_tokens": data.get("max_new_tokens", 150),
        "temperature":    data.get("temperature", 0.7),
        "top_p":          data.get("top_p", 0.9),
        "repetition_penalty": data.get("repetition_penalty", 1.2),
    }

    try:
        answer = ask(question, **params)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, port=5000)
