from trainer import TextClassifier
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
from data_tokenizer import LabelDataset


class ActiveLearningClassifier(TextClassifier):
    def __init__(self, model_name, num_labels, train_data_path, device=None, learning_rate=3e-5, batch_size=4,
                 num_epochs=5, confidence_threshold=0.9):
        super().__init__(model_name, num_labels, train_data_path, device, learning_rate, batch_size, num_epochs)
        self.confidence_threshold = confidence_threshold

    def label_data(self, unlabeled_data_path):
        unlabeled_dataset = self.tokenizer.get_unlabeled_data(unlabeled_data_path)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        labeled_data = []
        uncertain_data = []

        with torch.no_grad():
            for batch in tqdm(unlabeled_loader, desc="Labeling Data"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                max_probs, preds = torch.max(probs, dim=1)
                for i, prob in enumerate(max_probs):
                    if prob.item() < self.confidence_threshold:
                        uncertain_data.append({
                            'input_ids': input_ids[i].cpu().numpy(),
                            'attention_mask': attention_mask[i].cpu().numpy(),
                            'pred': preds[i].item(),
                            'confidence': prob.item()
                        })
                    else:
                        labeled_data.append({
                            'input_ids': input_ids[i].cpu().numpy(),
                            'attention_mask': attention_mask[i].cpu().numpy(),
                            'label': preds[i].item()
                        })

        return labeled_data, uncertain_data

    def train_on_labeled_data(self, labeled_data):
        # Convert labeled_data to dataset format suitable for DataLoader
        train_dataset = LabelDataset(labeled_data)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Train the model on the newly labeled data
        self.train()

    def refine_with_human_labels(self, uncertain_data, human_labels):
        # Combine uncertain data with human labels
        for i, data in enumerate(uncertain_data):
            data['label'] = human_labels[i]

        # Add the refined data to the existing training dataset
        refined_data = self.tokenizer.combine_with_existing_data(uncertain_data)
        self.train_on_labeled_data(refined_data)


if __name__ == '__main__':
    active_classifier = ActiveLearningClassifier(
        model_name="bert-base-multilingual-cased",
        num_labels=4,
        train_data_path='data/label_data.csv'
    )

    # Load the pre-trained model
    active_classifier.load_model('model/24_05_23_v1_54_saved_model.pth')

    labeled_data, uncertain_data = active_classifier.label_data('data/unlabeled_data_0_to_200.csv')
    print(f"Labeled Data: {len(labeled_data)}")
    print(f"Uncertain Data: {len(uncertain_data)}")

    # Save labeled and uncertain data to CSV files
    active_classifier.save_data_to_csv(labeled_data, 'data/result/labeled_data.csv')
    active_classifier.save_data_to_csv(uncertain_data, 'data/result/uncertain_data.csv', is_uncertain_data=True)
