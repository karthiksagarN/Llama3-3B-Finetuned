from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import difflib

# -------------------------
#  Loading Config, Model & Pipeline
# -------------------------
# Hugging Face token (needed if model is private/gated)
HF_TOKEN = os.getenv("HF_TOKEN")

# Your fine-tuned model on the Hub
model_id = "karthiksagarn/llama3-3.2b-finetuned-financial"

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    use_auth_token=HF_TOKEN,
    device_map="auto"
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

labels = [
    "Food & Drinks",
    "Travel & Transport",
    "Shopping",
    "Bills & Utilities",
    "Entertainment",
    "Health & Fitness",
    "Groceries",
    "Education",
    "Income",
    "Investments",
    "Withdrawals",
    "Miscellaneous"
]

# Few-shot examples
FEW_SHOT_EXAMPLES = [
    ("Sent Rs.510.00 From HDFC Bank A/C *0552 To Swiggy Limited", "Food & Drinks"),
    ("Salary credited from ACME Technologies", "Income"),
    ("Paid electricity bill online", "Bills & Utilities"),
    ("Bought vegetables at the local market", "Groceries"),
    ("ATM cash withdrawal 3000", "Withdrawals"),
]
examples_text = "\n\n".join(f"Description: {d}\nCategory: {c}" for d, c in FEW_SHOT_EXAMPLES)

# -------------------------
#  FastAPI App
# -------------------------
app = FastAPI()

class Transaction(BaseModel):
    description: str

@app.get("/")
def home():
    return {"message": "Bank Transaction Classifier is running 🚀"}

@app.post("/predict")
def predict(tx: Transaction):
    prompt = (
        "You are a classifier. Return EXACTLY one of the labels below and NOTHING ELSE.\n\n"
        f"Labels: {', '.join(labels)}\n\n"
        "Examples:\n"
        f"{examples_text}\n\n"
        "Now classify the following:\n"
        f"Description: {tx.description}\n"
        "Category:"
    )

    out = pipe(prompt, max_new_tokens=8, do_sample=False, temperature=0.0)
    generated = out[0]["generated_text"].split("Category:")[-1].strip()
    generated = generated.split("\n")[0].strip()

    # Post-processing
    if generated in labels:
        return {"predicted_category": generated}

    for lbl in labels:
        if lbl.lower() in generated.lower() or generated.lower() in lbl.lower():
            return {"predicted_category": lbl}

    match = difflib.get_close_matches(generated, labels, n=1, cutoff=0.55)
    if match:
        return {"predicted_category": match[0]}

    return {"predicted_category": "Miscellaneous"}
