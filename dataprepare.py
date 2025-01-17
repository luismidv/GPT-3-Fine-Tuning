#SCRIPT TO PREPARE DATASET FOR A CHATBOT AI
import pandas as pd
from transformers import AutoTokenizer, TrainingArguments, Trainer, GPT2ForSequenceClassification
from transformers import GPT2Model
from datasets import load_dataset
import evaluate
import numpy as np
import sklearn


def function_tokenizer(features):
    #USING padding = "max_length fails with gpt2."
    inputs = autotokenizer(features['text'], truncation = True, max_length = 128)
    inputs['labels'] = inputs['input_ids'].copy()
    return inputs

def calculate_metrics(eval_pred):
    logits,labels = eval_pred
    predictions = np.argmax(logits,axis = 1)
    return metric.compute(predictions = predictions, references = labels)


dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
train_dataset = dataset['train']
print(dataset)

autotokenizer = AutoTokenizer.from_pretrained("gpt2")
autotokenizer.pad_token = autotokenizer.eos_token

tokenized = train_dataset.map(function_tokenizer)
model = GPT2ForSequenceClassification.from_pretrained("gpt2")

training_args = TrainingArguments(output_dir = "test_trainer", eval_strategy = "epoch")
metric = evaluate.load("accuracy")

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset=tokenized,
    eval_dataset=tokenized,
    compute_metrics=calculate_metrics

)
trainer.train()



