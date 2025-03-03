import streamlit as st
from transformers import pipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Streamlit app
st.title("AI News Article Writer")
st.write("Enter a headline to generate a news article.")

news_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_article(headline, max_length=200):
    prompt = f"Headline: {headline}\nArticle:"
    article = news_generator(prompt, max_length=max_length, num_return_sequences=1)
    return article[0]["generated_text"]

# Input
headline = st.text_input("Headline:")

# Generate article
if headline:
    with st.spinner("Generating article..."):
        article = generate_article(headline)
        st.write(article)
