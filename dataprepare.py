#SCRIPT TO PREPARE DATASET FOR A CHATBOT AI
import pandas as pd
from datasets import load_dataset
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

def dataset_info(dataset):
    data = pd.read_csv(dataset)
    print(data.columns)
    print(data.describe())

print(dataset)

