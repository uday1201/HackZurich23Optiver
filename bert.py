import os

import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, TrainingArguments, Trainer
from torch.utils.data import DataLoader, TensorDataset, random_split

from transformers import BertModel
import torch.nn as nn

import wandb

class BertRegression(nn.Module):
    def __init__(self, num_outputs):
        super(BertRegression, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.regressor = nn.Linear(self.bert.config.hidden_size, num_outputs)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        return self.regressor(pooled_output)

use_wandb = True

# 1. Data Preparation
data = pd.read_csv('training.csv')
tweets = data['SocialMediaFeed'].tolist()
labels = data[['NVDA', 'ING', 'SAN', 'PFE', 'CSCO']].values.tolist()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. Preprocess the Data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded_data = tokenizer(tweets, truncation=True, padding=True, max_length=256, return_tensors="pt")
input_ids = encoded_data['input_ids'].to(device)
attention_masks = encoded_data['attention_mask'].to(device)
labels = torch.tensor(labels).to(device)
labels *= 100

dataset = TensorDataset(input_ids, attention_masks, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
val_dataloader = DataLoader(val_dataset, batch_size=8)

# 3. Model Setup
model = BertRegression(num_outputs=5).to(device)

# 4. Training
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.MSELoss()

if use_wandb:
    wandb.init(project="stock-prediction")
    wandb.watch(model)

MODEL_DIR = 'model_checkpoints'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

epochs = 10
for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        input_ids, attention_mask, batch_labels = batch
        
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, batch_labels)
        print("Epoch:", str(epoch), "-- Training Loss:", str(loss.item()))
        if use_wandb:
            wandb.log({"Training Loss": loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, batch_labels = batch
            
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, batch_labels)
            val_loss += loss.item()
            print("Epoch:", str(epoch), "-- Validation Loss:", str(loss.item()))
    
    print(f"Epoch: {epoch+1}, Validation Loss: {val_loss/len(val_dataloader)}")
    if use_wandb:
        wandb.log({"Validation Loss": val_loss/len(val_dataloader)})
    
    model_checkpoint_path = os.path.join(MODEL_DIR, f'epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), model_checkpoint_path)
    if use_wandb:
        wandb.save(model_checkpoint_path)