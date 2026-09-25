!pip install "transformers[torch]"
import pandas as pd
from transformers import T5Tokenizer , trainer,TrainArguments,T5ForConditionalGeneration
train_data=pd.read_csv("samsum-train.csv")
val_data=pd.read_csv("samsum-validation.csv")
train_data.head()
train_data.shape
val_data.shape
#random sampling
train_data=train_data.sample(n=4000,random_state=42).reset_index(drop=True)
val_data=val_data.sample(n=500,random_state=42).reset_index(drop=True)
train_data.shape
#datapre processing
import re

def clean_data(text):
    tex=re.sub(r"\r\n"," ",text)
    tex=re.sub(r"\s+"," ",text)
    tex=re.sub(r"<.*?>"," ",text)
    text=text.strip().lower()
    return text
  train_data["dialogue"]=train_data["dialogue"].apply(clean_data)
train_data["summary"]=train_data["summary"].apply(clean_data)
val_data["dialogue"]=val_data["dialogue"].apply(clean_data)
val_data["summary"]=val_data["summary"].apply(clean_data)
#tocanization give id for each word
tokenizer=T5Tokenizer.from_pretrained("t5-small")
#raw data convert to tokens
def tokenize(data):
    tokenizer
