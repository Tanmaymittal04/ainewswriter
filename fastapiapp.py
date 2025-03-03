from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import os

# Create the FastAPI application instance
app = FastAPI()

# Set the model path
model_path = "./miniforge3"  # Replace with your actual path

# Verify the path exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model directory not found: {model_path}")

# Load the model and tokenizer
try:
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
except Exception as e:
    raise OSError(f"Error loading model or tokenizer: {e}")

# Initialize the pipeline with the loaded model and tokenizer
news_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Define the request body schema
class HeadlineRequest(BaseModel):
    headline: str

# Define the generate_article function
def generate_article(headline, max_length=200):
    prompt = f"Headline: {headline}\nArticle:"
    article = news_generator(prompt, max_length=max_length, num_return_sequences=1)
    return article[0]["generated_text"]

# Define the API endpoint
@app.post("/generate-article")
def generate_article_api(request: HeadlineRequest):
    article = generate_article(request.headline)
    return {"article": article}
