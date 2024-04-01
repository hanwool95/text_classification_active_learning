import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AdamW
from data_tokenizer import DataTokenizer
from tqdm.auto import tqdm

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


if __name__ == '__main__':
    tokenizer = DataTokenizer("bert-base-multilingual-cased")
    train_dataset, val_dataset = tokenizer.get_train_test_split_data('data/label_data.csv')

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=4).to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    train(model, train_loader, optimizer)
