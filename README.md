# Light-weight-fine-tuning


Lightweight Fine-Tuning
This repository contains the implementation of a lightweight fine-tuning approach for pre-trained models using PyTorch and Hugging Face's Transformers library. The script LightweightFineTuning.py is designed to perform sequence classification tasks with the distilbert-base-uncased model.

Table of Contents
Installation
Usage
Dataset
Training
Evaluation
Results
Installation

To run the script, you need to have Python installed along with the required libraries. You can install the dependencies using pip:

bash
Copy code
pip install -r requirements.txt
Make sure your requirements.txt includes:

Copy code
torch
transformers
datasets
numpy
scikit-learn
Usage
Running the Script
To run the fine-tuning script, use the following command:

bash
Copy code
python LightweightFineTuning.py
Arguments
The script accepts the following arguments:

--dataset_path: Path to the dataset.
--model_name: Name of the pre-trained model to be used (default is distilbert-base-uncased).
--epochs: Number of training epochs (default is 3).
--batch_size: Batch size for training (default is 16).
--learning_rate: Learning rate for the optimizer (default is 5e-5).
Example usage:

bash
Copy code
python LightweightFineTuning.py --dataset_path ./data/dataset.csv --epochs 5 --batch_size 32
Dataset
The script expects a dataset in CSV format with at least two columns: text and label. The text column should contain the text samples, and the label column should contain the corresponding labels.

Training
The script performs the following steps during training:

Loads the dataset and preprocesses the text and labels.
Tokenizes the text using the tokenizer from the pre-trained model.
Initializes the distilbert-base-uncased model for sequence classification.
Trains the model using the specified number of epochs and batch size.
Saves the trained model.
Evaluation
After training, the script evaluates the model on a test set and prints the evaluation metrics such as accuracy, precision, recall, and F1-score.

Results
The results of the fine-tuning process, including the evaluation metrics, will be printed to the console. Additionally, the trained model will be saved to the specified directory.

