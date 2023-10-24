import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset

# Specify the column names based on the dataset specification
column_names = ["polarity", "tweet_id", "tweet_date", "query", "user", "text"]

# Read the CSV with the specified column names
data = pd.read_csv("./Data/training.1600000.processed.noemoticon.csv", encoding='ISO-8859-1', names=column_names)

# Assuming you have a DataFrame 'data' with a 'text' column for text and 'polarity' column for labels
train_texts = data['text'].tolist()
train_labels = data['polarity'].tolist()

# Split data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42
)

# Define a custom dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# Define a tokenizer and the maximum sequence length
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_length = 64

# Create custom datasets for training and testing
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length)
test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_length)

# Define data loaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32)

# Initialize the model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=3
)  # 3 classes: negative, neutral, positive

# Define training parameters (e.g., optimizer and loss function)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Train the model (loop through epochs, batches, and backpropagation)
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = model(input_ids, attention_mask=attention_mask)
        predicted_labels = torch.argmax(outputs.logits, dim=1)

        predictions.extend(predicted_labels.tolist())
        true_labels.extend(labels.tolist())

# Map the labels back to the original classes (0 = negative, 2 = neutral, 4 = positive)
original_labels = {0: 0, 1: 2, 2: 4}
sentiment_predictions = [original_labels[label] for label in predictions]

# Calculate and print classification report
report = classification_report(true_labels, sentiment_predictions)
print(report)
