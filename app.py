from flask import Flask, render_template, request, session, redirect, url_for
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)
app.secret_key = 'mira_secret_key_123' 


model_name = "tiiuae/falcon-rw-1b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route("/", methods=["GET", "POST"])
def index():
    if "chat" not in session:
        session["chat"] = []

    chat = session["chat"]

    if request.method == "POST":
        user_input = request.form["prompt"]
        chat.append({"role": "user", "text": user_input})

        inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
        outputs = model.generate(inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        ai_reply = response.replace(user_input, "").strip()
        chat.append({"role": "ai", "text": ai_reply})


        session["chat"] = chat  

        return redirect(url_for("index"))  

    return render_template("index.html", messages=chat)

if __name__=="__main__":
    app.run(debug=True)