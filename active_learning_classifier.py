from trainer import TextClassifier
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
from data_tokenizer import LabelDataset
import pandas as pd

class ActiveLearningClassifier(TextClassifier):
    def __init__(self, model_name, num_labels, train_data_path, device=None, learning_rate=3e-5, batch_size=4, num_epochs=5, confidence_threshold=0.9):
        super().__init__(model_name, num_labels, train_data_path, device, learning_rate, batch_size, num_epochs)
        self.confidence_threshold = confidence_threshold

    def label_data(self, unlabeled_data_path):
        unlabeled_dataset, original_indices = self.tokenizer.get_unlabeled_data_with_indices(unlabeled_data_path)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        labeled_data = []
        uncertain_data = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(unlabeled_loader, desc="Labeling Data")):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                max_probs, preds = torch.max(probs, dim=1)
                batch_indices = original_indices[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
                for i, prob in enumerate(max_probs):
                    data_point = {
                        'input_ids': input_ids[i].cpu().numpy(),
                        'attention_mask': attention_mask[i].cpu().numpy(),
                        'index': batch_indices[i]
                    }
                    if prob.item() < self.confidence_threshold:
                        data_point.update({
                            'pred': preds[i].item(),
                            'confidence': prob.item()
                        })
                        uncertain_data.append(data_point)
                    else:
                        data_point.update({
                            'label': preds[i].item()
                        })
                        labeled_data.append(data_point)

        return labeled_data, uncertain_data

    def train_on_labeled_data(self, labeled_data):
        encodings = {key: [d[key] for d in labeled_data] for key in labeled_data[0].keys() if key not in ['label', 'index']}
        labels = [d['label'] for d in labeled_data]
        train_dataset = LabelDataset(encodings, labels)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.train()

    def refine_with_human_labels(self, uncertain_data, labeled_data):
        human_labeled_data = []
        for data in uncertain_data:
            text = self.tokenizer.decode(data['input_ids'], skip_special_tokens=True)
            while True:
                try:
                    print(f"Text: {text}")
                    label = int(input("Enter the correct label: "))
                    data['label'] = label
                    human_labeled_data.append(data)
                    break
                except ValueError:
                    print("Invalid input. Please enter a valid integer label.")

        # Combine the new human labeled data with the existing labeled data
        refined_data = labeled_data + human_labeled_data

        # Save refined data to CSV
        self.save_combined_data_to_csv(refined_data, 'data/result/combined_labeled_data.csv')

        self.train_on_labeled_data(refined_data)


if __name__ == '__main__':
    active_classifier = ActiveLearningClassifier(
        model_name="bert-base-multilingual-cased",
        num_labels=5,
        train_data_path='data/label_data.csv',
        confidence_threshold=0.5,
    )

    # Load the pre-trained model
    active_classifier.load_model('model/saved_model.pth')

    # Label the data and get uncertain data
    labeled_data, uncertain_data = active_classifier.label_data('data/unlabeled_data_0_to_200.csv')
    print(f"Labeled Data: {len(labeled_data)}")
    print(f"Uncertain Data: {len(uncertain_data)}")

    # Save labeled and uncertain data to CSV files
    active_classifier.save_data_to_csv(labeled_data, 'data/result/labeled_data.csv')
    active_classifier.save_data_to_csv(uncertain_data, 'data/result/uncertain_data.csv', is_uncertain_data=True)

    # Refine with human labels and train the model again
    active_classifier.refine_with_human_labels(uncertain_data, labeled_data)

    # Evaluate the refined model
    val_loss, accuracy, f1 = active_classifier.evaluate()
    print(f"Validation Loss: {val_loss}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")

    # Save the refined model
    active_classifier.save_model('model/refined_model.pth')
