import streamlit as st
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from src.pipeline.prediction import PredictionPipeline

@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained('maichmarc/textS')
    model = AutoModelForSeq2SeqLM.from_pretrained("maichmarc/textS")
    return tokenizer,model

st.title('Text Summariser')
st.write("This is Text Summarisation model that uses the google/pegasus-cnn_dailymail model fine-tuned using the Samsung/samsum dataset" \
"using Huuging Face's transformers. Evaluation was achieved using ROUGE metrics.")
tokenizer,model = get_model()

user_input = st.text_area(
    'Enter Text to Summarize',
    height=400)
button = st.button("Summarise")


if user_input and button :
 
    obj = PredictionPipeline()
    text = obj.predict(user_input)

    st.write("Summary: ", text)