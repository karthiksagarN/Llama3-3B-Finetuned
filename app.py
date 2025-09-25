from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# -----------------------
# 1. Define categories
# -----------------------
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

# -----------------------
# 2. Load model + tokenizer
# -----------------------
pipe = pipeline(
    "text-generation",
    model="./saved_model",     # your trained model path
    tokenizer="./saved_model",
    device_map="auto"
)

# -----------------------
# 3. FastAPI app
# -----------------------
app = FastAPI(title="Bank Transaction Classifier")

class TransactionRequest(BaseModel):
    text: str

class TransactionResponse(BaseModel):
    category: str

# -----------------------
# 4. Inference function
# -----------------------
def classify_transaction(text: str) -> str:
    prompt = (
        f"Classify the following bank transaction into one of these categories:\n"
        f"{', '.join(labels)}\n\n"
        f"Description: {text}\n\nCategory:"
    )
    output = pipe(prompt, max_new_tokens=20, do_sample=False)
    generated = output[0]["generated_text"].split("Category:")[-1].strip()

    # take only first line
    generated = generated.split("\n")[0].strip()

    # # Ensure it matches one of the labels
    # for label in labels:
    #     if label.lower() in generated.lower():
    #         return label

    # return "Miscellaneous"     ----- we can use this if we need a fallback

    return generated

# -----------------------
# 5. API endpoint
# -----------------------
@app.post("/classify", response_model=TransactionResponse)
def classify(req: TransactionRequest):
    category = classify_transaction(req.text)
    return {"text":req.text, "category": category}
