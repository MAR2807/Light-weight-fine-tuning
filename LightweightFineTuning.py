#!/usr/bin/env python
# coding: utf-8

# # Lightweight Fine-Tuning Project

# ## Loading and Evaluating a Foundation Model
# 



from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "bert-base-uncased"


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)




dataset = load_dataset("yelp_polarity", trust_remote_code=True)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=128)

train_dataset = dataset["train"].select(range(50)).rename_column('label','labels').map(preprocess_function, batched=True)
test_dataset = dataset["test"].select(range(100)).rename_column('label','labels').map(preprocess_function, batched=True)

class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        inputs = self.tokenizer(item['text'], max_length=128, truncation=True, padding='max_length', return_tensors="pt")
        inputs['labels'] = torch.tensor(item['labels'])
        return {key: torch.squeeze(value) for key, value in inputs.items()}

train_dataset_loader = DataLoader(CustomDataset(train_dataset, tokenizer), batch_size=8)
test_dataset_loader = DataLoader(CustomDataset(test_dataset, tokenizer), batch_size=8)






def evaluate(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs = {key: value.to(device) for key, value in batch.items() if key != 'labels'}
            labels = batch['labels'].to(device)
            
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2f}")

# Call the evaluate function with the training dataset
evaluate(model, train_dataset_loader)




# ## Performing Parameter-Efficient Fine-Tuning
# 
# TODO: In the cells below, create a PEFT model from your loaded model, run a training loop, and save the PEFT model weights.

# In[3]:


# Define and initialize LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.2,
    task_type=TaskType.SEQ_CLS,
)

model2 = get_peft_model(model, lora_config, "default")
print(f"Type of model2 is {type(model2)}")




training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
)





trainer = Trainer(
    model=model2,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)





trainer.train()





# Save model
torch.save(model2.state_dict(), "model2_model.safetensors")


# ## Performing Inference with a PEFT Model
# 



model2_reinit = get_peft_model(model, lora_config)
model2_reinit.load_state_dict(torch.load("model2_model.safetensors"))

print(f"Type of model2_reinit is {type(model2_reinit)}")

try:
    model2_reinit.eval()
    print("Model 2 reinit is in eval mode")
except AttributeError as e:
    print(f"Error setting model2 to eval mode: {e}")





evaluate(model, test_dataset_loader)
evaluate(model2_reinit, test_dataset_loader)















