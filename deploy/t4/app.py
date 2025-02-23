import torch
from flask import Flask, request, render_template
from transformers import BertTokenizer
from model import SBERT

# Initialize Flask app
app = Flask(__name__)

# Load tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")
model = SBERT().to(device)
model.load_state_dict(torch.load("../sbert_task2.pth", map_location=device))
model.eval()

# Mapping of label indices
label_map = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

def predict_nli(premise, hypothesis):
    """Tokenize input and predict label."""
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
    input_ids, attention_mask = inputs["input_ids"].to(device), inputs["attention_mask"].to(device)

    with torch.no_grad():
        mid_point = input_ids.shape[1] // 2
        input_ids_a, input_ids_b = input_ids[:, :mid_point], input_ids[:, mid_point:]
        attention_mask_a, attention_mask_b = attention_mask[:, :mid_point], attention_mask[:, mid_point:]

        logits = model(input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
        pred_label = torch.argmax(logits, dim=1).item()

    return label_map[pred_label]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        premise = request.form["premise"]
        hypothesis = request.form["hypothesis"]
        prediction = predict_nli(premise, hypothesis)
        return render_template("index.html", premise=premise, hypothesis=hypothesis, prediction=prediction)

    return render_template("index.html", premise="", hypothesis="", prediction="")

if __name__ == "__main__":
    app.run(debug=True)
