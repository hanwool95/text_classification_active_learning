import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AdamW
from data_tokenizer import DataTokenizer
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, data_loader, optimizer):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    print("Training start")

    progress_bar = tqdm(data_loader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix({'loss': loss.item()})

def evaluate(model, data_loader):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    val_loss = 0
    preds = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            val_loss += loss.item()
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    val_loss /= len(data_loader)
    accuracy = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='weighted')

    return val_loss, accuracy, f1

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

if __name__ == '__main__':
    tokenizer = DataTokenizer("bert-base-multilingual-cased")
    train_dataset, val_dataset = tokenizer.get_train_test_split_data('data/label_data.csv')

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=4).to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    train(model, train_loader, optimizer)

    val_loss, accuracy, f1 = evaluate(model, val_loader)
    print(f"Validation Loss: {val_loss}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")

    save_model(model, 'model/saved_model.pth')
