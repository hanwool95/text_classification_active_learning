from trainer import TextClassifier

if __name__ == '__main__':
    classifier = TextClassifier(
        model_name="bert-base-multilingual-cased",
        num_labels=5,
        train_data_path='data/label_data_v3.csv'
    )

    # Load the pre-trained model
    classifier.load_model('model/24_05_25_69_until600_total.pth')

    # Evaluate the refined model
    val_loss, accuracy, f1 = classifier.evaluate()
    print(f"Validation Loss: {val_loss}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")