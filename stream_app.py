import streamlit as st
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from src.pipeline.prediction import PredictionPipeline

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = AutoTokenizer.from_pretrained('maichmarc/textS')
    model = AutoModelForSeq2SeqLM.from_pretrained("maichmarc/textS")
    return tokenizer,model


tokenizer,model = get_model()

user_input = st.text_area('Enter Text to Analyze')
button = st.button("Summarise")

# d = {
    
#   1:'Toxic',
#   0:'Non Toxic'
# }

if user_input and button :
    # test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
    # test_sample
    # output = model(**test_sample)
    # st.write("Logits: ",output.logits)
    # y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    obj = PredictionPipeline()
    text = obj.predict(user_input)

    st.write("Prediction: \n", text)#d[y_pred[0]])