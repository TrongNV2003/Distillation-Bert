# Distillation for multi-intent-classification task
Predict multiple intents for each message on conversations

## Installation
```sh
pip install -r requirements.txt
```

## Usage
training and evaluating models:
```sh
bash train.sh
```

## Result
Teacher model: phobert-large
Student model: phobert-base, phobert-base-v2

| Models                 | Accuracy    | Precision   | Recall      | F1 Score     |
|----------------------- |:-----------:|:-----------:|:-----------:|:------------:|
| DistilPhoBert base     | 73.84       | 79.84       | 75.15       | 75.99        |
| DistilPhoBert-base-v2  | 77.91       | 86.09       | 79.76       | 82.00        |

