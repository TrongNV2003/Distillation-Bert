import random
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)

from distil_multi_intent_classification.services.evaluate import Tester
from distil_multi_intent_classification.services.trainer import LlmTrainer
from distil_multi_intent_classification.services.dataloader import Dataset, LlmDataCollator

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def count_parameters(model: torch.nn.Module) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

parser = argparse.ArgumentParser()
parser.add_argument("--dataloader_workers", type=int, default=2, required=True)
parser.add_argument("--device", type=str, default="cuda", required=True)
parser.add_argument("--seed", type=int, default=42, required=True)
parser.add_argument("--epochs", type=int, default=10, required=True)
parser.add_argument("--teacher_finetune_epochs", type=int, default=5, required=True)
parser.add_argument("--learning_rate", type=float, default=3e-5, required=True)
parser.add_argument("--teacher_learning_rate", type=float, default=2e-5, required=True)
parser.add_argument("--weight_decay", type=float, default=0.01, required=True)
parser.add_argument("--max_length", type=int, default=256, required=True)
parser.add_argument("--pad_mask_id", type=int, default=-100, required=True)
parser.add_argument("--teacher_model", type=str, default="vinai/phobert-large", required=True)
parser.add_argument("--student_model", type=str, default="vinai/phobert-base-v2", required=True)
parser.add_argument("--train_batch_size", type=int, default=16, required=True)
parser.add_argument("--valid_batch_size", type=int, default=16, required=True)
parser.add_argument("--test_batch_size", type=int, default=16, required=True)
parser.add_argument("--warmup_steps", type=int, default=100, required=True)
parser.add_argument("--train_file", type=str, default="dataset/train.json", required=True)
parser.add_argument("--valid_file", type=str, default="dataset/val.json", required=True)
parser.add_argument("--test_file", type=str, default="dataset/test.json", required=True)
parser.add_argument("--output_dir", type=str, default="./models/classification", required=True)
parser.add_argument("--record_output_file", type=str, default="output.json")
parser.add_argument("--evaluate_on_accuracy", type=bool, default=True, required=True)
parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=False)
parser.add_argument("--early_stopping_patience", type=int, default=3, help="Number of epochs to wait for improvement", required=True)
parser.add_argument("--early_stopping_threshold", type=float, default=0.001, help="Minimum improvement to reset early stopping counter", required=True)
args = parser.parse_args()

def get_tokenizer(checkpoint: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    """
    Input format: <history>{history_1}<sep>{history_2}<sep>...<sep>{history_n}</history><current>{context}</current>
    """
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<history>', '</history>', '<current>', '</current>']}
    )
    return tokenizer

def get_teacher_model(checkpoint: str, device: str, tokenizer: AutoTokenizer) -> AutoModel:
    model = AutoModel.from_pretrained(checkpoint)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    return model

def get_student_model(
        checkpoint: str, device: str, tokenizer: AutoTokenizer, num_labels: str, id2label: dict, label2id: dict
    ) -> AutoModelForSequenceClassification:
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        problem_type="multi_label_classification",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    return model


if __name__ == "__main__":
    set_seed(args.seed)

    unique_labels = ["Cung cấp thông tin", "Tương tác", "Hỏi thông tin giao hàng", "Hỗ trợ, hướng dẫn", "Yêu cầu", "Phản hồi", "Sự vụ", "UNKNOWN"]
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for idx, label in enumerate(unique_labels)}

    tokenizer = get_tokenizer(args.teacher_model)
    
    train_set = Dataset(json_file=args.train_file, tokenizer=tokenizer, label_mapping=label2id)
    valid_set = Dataset(json_file=args.valid_file, tokenizer=tokenizer, label_mapping=label2id)
    test_set = Dataset(json_file=args.test_file, tokenizer=tokenizer, label_mapping=label2id)

    collator = LlmDataCollator(tokenizer=tokenizer, max_length=args.max_length)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    teacher = get_teacher_model(args.teacher_model, device, tokenizer)
    student = get_student_model(args.student_model, device, tokenizer, num_labels=len(unique_labels), id2label=id2label, label2id=label2id)

    print(f"\nLabel: {unique_labels}")
    print(f"\nTeacher: {args.teacher_model}")
    count_parameters(teacher)
    print(f"Student: {args.student_model}")
    count_parameters(student)

    model_name = args.student_model.split('/')[-1]
    save_dir = f"{args.output_dir}-{model_name}"

    trainer = LlmTrainer(
        dataloader_workers=args.dataloader_workers,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        teacher_learning_rate=args.teacher_learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        teacher_model=teacher,
        student_model=student,
        pin_memory=args.pin_memory,
        save_dir=save_dir,
        tokenizer=tokenizer,
        train_set=train_set,
        valid_set=valid_set,
        train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size,
        collator_fn=collator,
        evaluate_on_accuracy=args.evaluate_on_accuracy,
        teacher_finetune_epochs=args.teacher_finetune_epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
    )
    trainer.train()


    # Test model on test set
    tuned_model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(device)
    teacher_model = AutoModel.from_pretrained(f"{save_dir}/teacher").to(device)
    teacher_classifier = nn.Linear(
        teacher_model.config.hidden_size, 
        len(unique_labels), 
        dtype=torch.float32
    ).to(device)
    teacher_classifier.load_state_dict(torch.load(f"{save_dir}/teacher_classifier.pt"))

    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, collate_fn=collator)
    
    tester = Tester(
        model=tuned_model,
        teacher_model=teacher_model,
        teacher_classifier=trainer.teacher_classifier,
        test_loader=test_loader,
        output_file=args.record_output_file
    )

    tester.evaluate()

    print(f"\nmodel: {args.student_model}")
    print(f"Teacher: {args.teacher_model}")
    print(f"params: lr {args.learning_rate}, epoch {args.epochs}")

