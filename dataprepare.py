#SCRIPT TO PREPARE DATASET FOR A CHATBOT AI
import pandas as pd
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import GPT2Model
from datasets import load_dataset
import evaluate

dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")


def dataset_info(dataset):
    data = pd.read_csv(dataset)
    print(data.columns)
    print(data.describe())

def function_tokenizer(features):
    #USING padding = "max_length fails with gpt2."
    return autotokenizer(features['instruction'],  truncation = True)

def calculate_metrics(eval_pred):
    logits,labels = eval_pred
    predictions = np.argmax(logits,axis = 1)
    return metric.compute(predictions = predictions, references = labels)

print(dataset)
autotokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenized = dataset.map(function_tokenizer, batched = True)
model = GPT2Model.from_pretrained("EleutherAI/gpt-neo-1.3")

training_args = TrainingArguments(output_dir = "test_trainer", eval_strategy = "epoch")
metric.evaluate.load("accuracy")

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset=tokenized,
    eval_dataset=tokenized,
    compute_metrics=calculate_metrics

)

trainer.train()


