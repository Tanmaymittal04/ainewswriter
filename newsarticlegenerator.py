from transformers import pipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Load fine-tuned model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
news_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_article(headline, max_length=200):
    prompt = f"Headline: {headline}\nArticle:"
    article = news_generator(prompt, max_length=max_length, num_return_sequences=1)
    return article[0]["generated_text"]

# Example usage
headline = "AI Revolutionizes Healthcare"
article = generate_article(headline)
print(article)
