import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AdamW
from data_tokenizer import DataTokenizer
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataTokenizerExtended(DataTokenizer):
    def get_tokenized_data_with_labels(self, data_url):
        df = pd.read_csv(data_url)
        df['text'] = df['Messages'].fillna('') + " [PAD] " + df['Remain Messages'].fillna('')
        labels = df['label'].tolist()
        encodings = self.tokenize_function(df['text'].tolist())
        return encodings, labels


def train(model, data_loader, optimizer):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    print("training start")
    for index, batch in enumerate(data_loader):
        print(str(index/len(data_loader)*100)+"%")
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    tokenizer = DataTokenizerExtended("bert-base-multilingual-cased")
    train_dataset, val_dataset = tokenizer.get_train_test_split_data('data/label_data.csv')

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=4).to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    train(model, train_loader, optimizer)
