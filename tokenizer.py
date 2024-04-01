import pandas as pd
from transformers import AutoTokenizer


class DataTokenizer:
    def __init__(self, model_name: str):
        """
        Constructor for the DataTokenizer class.
        :param model_name: Name of the model to be used.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(self, examples):
        """
        Function to tokenize the given examples.
        :param examples: A list of texts to be tokenized.
        :return: Tokenized output.
        """
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def get_tokenized_data(self, data_url: str):
        """
        Loads data from a CSV file and tokenizes it.
        :param data_url: Path to the data file.
        :return: Tokenized data.
        """
        df = pd.read_csv(data_url)
        df['text'] = df['Messages'].fillna('') + " " + df['Remain Messages'].fillna('')
        prepared_data = {"text": df['text'].tolist()}
        return self.tokenize_function(prepared_data)


if __name__ == '__main__':
    tokenizer = DataTokenizer("bert-base-multilingual-cased")
    tokenized_data = tokenizer.get_tokenized_data('data.csv')
    print(tokenized_data['input_ids'][0])
