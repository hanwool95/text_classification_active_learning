import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch


class LabelDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


class UnlabeledDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])


class DataTokenizer:
    def __init__(self, model_name: str):
        """
        Constructor for the DataTokenizer class.
        :param model_name: Name of the model to be used.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token='[PAD]')

    def tokenize_function(self, texts):
        """
        Function to tokenize the given examples.
        :param texts: A list of texts to be tokenized.
        :return: Tokenized output.
        """
        return self.tokenizer(texts, padding="max_length", truncation=True)

    def decode(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_tokenized_data(self, data_url: str):
        """
        Loads data from a CSV file and tokenizes it.
        :param data_url: Path to the data file.
        :return: Tokenized data.
        """
        df = pd.read_csv(data_url)
        df['Messages'] = df['Messages'].fillna('')  # NaN 값을 빈 문자열로 대체
        df['Remain Messages'] = df['Remain Messages'].fillna('')  # NaN 값을 빈 문자열로 대체
        df['text'] = df['Messages'] + " [PAD] " + df['Remain Messages']
        prepared_data = df['text'].tolist()
        return self.tokenize_function(prepared_data)

    def get_train_test_split_data(self, data_url: str, test_size: int = 0.5):
        df = pd.read_csv(data_url)
        df['Messages'] = df['Messages'].fillna('')  # NaN 값을 빈 문자열로 대체
        df['Remain Messages'] = df['Remain Messages'].fillna('')  # NaN 값을 빈 문자열로 대체
        df['text'] = df['Messages'] + " [PAD] " + df['Remain Messages']

        # NaN 값을 가진 레이블 제거
        df = df.dropna(subset=['label'])

        texts = df['text'].tolist()
        labels = df['label'].tolist()

        train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=test_size,
                                                                              random_state=42)

        train_encodings = self.tokenize_function(train_texts)
        test_encodings = self.tokenize_function(test_texts)

        train_dataset = LabelDataset(train_encodings, train_labels)
        test_dataset = LabelDataset(test_encodings, test_labels)

        return train_dataset, test_dataset

    def get_unlabeled_data(self, data_url: str):
        df = pd.read_csv(data_url)
        df['Messages'] = df['Messages'].fillna('')
        df['Remain Messages'] = df['Remain Messages'].fillna('')
        df['text'] = df['Messages'] + " [PAD] " + df['Remain Messages']

        texts = df['text'].tolist()
        encodings = self.tokenize_function(texts)

        return UnlabeledDataset(encodings)


if __name__ == '__main__':
    tokenizer = DataTokenizer("bert-base-multilingual-cased")
    tokenized_data = tokenizer.get_tokenized_data('data/label_data.csv')
    print(tokenized_data['input_ids'][0])
