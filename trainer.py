import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from data_tokenizer import DataTokenizer
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd


class TextClassifier:
    def __init__(self, model_name, num_labels, train_data_path, device=None, learning_rate=3e-5, batch_size=4,
                 num_epochs=5):
        self.model_name = model_name
        self.num_labels = num_labels
        self.train_data_path = train_data_path
        self.device = device if device else self.get_device()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.tokenizer = DataTokenizer(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels).to(
            self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.train_loader, self.val_loader = self.prepare_data()

    def get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def prepare_data(self):
        train_dataset, val_dataset = self.tokenizer.get_train_test_split_data(self.train_data_path)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def train(self):
        self.model.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        print("Training start")

        num_training_steps = len(self.train_loader) * self.num_epochs
        num_warmup_steps = int(0.1 * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)

        for epoch in range(self.num_epochs):
            progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
            for batch in progress_bar:
                self.optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                self.optimizer.step()
                scheduler.step()

                progress_bar.set_postfix({'loss': loss.item()})

    def evaluate(self):
        self.model.eval()
        loss_fn = torch.nn.CrossEntropyLoss()
        val_loss = 0
        preds = []
        true_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                val_loss += loss.item()
                preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        val_loss /= len(self.val_loader)
        accuracy = accuracy_score(true_labels, preds)
        f1 = f1_score(true_labels, preds, average='weighted')

        return val_loss, accuracy, f1

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {path}")

    def save_data_to_csv(self, data, path, is_uncertain_data=False):
        texts = self.decode_tokens(data)
        indices = [d['index'] for d in data]

        if is_uncertain_data:
            df = pd.DataFrame({
                'index': indices,
                'text': texts,
                'predicted_label': [d['pred'] for d in data],
                'confidence': [d['confidence'] for d in data]
            })
        else:
            df = pd.DataFrame({
                'index': indices,
                'text': texts,
                'label': [d['label'] for d in data]
            })
        df.sort_values(by='index', inplace=True)
        df.to_csv(path, index=False)
        print(f"Data saved to {path}")

    def save_combined_data_to_csv(self, data, path):
        texts = self.decode_tokens(data)
        indices = [d['index'] for d in data]
        df = pd.DataFrame({
            'index': indices,
            'text': texts,
            'label': [d['label'] for d in data]
        })
        df.sort_values(by='index', inplace=True)
        df.to_csv(path, index=False)
        print(f"Combined labeled data saved to {path}")

    def decode_tokens(self, encoded_data):
        return [self.tokenizer.decode(data['input_ids'], skip_special_tokens=True) for data in encoded_data]


if __name__ == '__main__':
    classifier = TextClassifier(model_name="bert-base-multilingual-cased", num_labels=5,
                                train_data_path='data/data_5label_v4.csv')
    classifier.train()
    val_loss, accuracy, f1 = classifier.evaluate()
    print(f"Validation Loss: {val_loss}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    classifier.save_model('model/saved_model.pth')
