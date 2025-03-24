import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score

import numpy as np
from tqdm import tqdm
from loguru import logger

from distil_multi_intent_classification.services.utils import AverageMeter

class LlmTrainer:
    def __init__(
        self,
        dataloader_workers: int,
        device: str,
        epochs: int,
        learning_rate: float,
        teacher_learning_rate: float,
        weight_decay: float,
        warmup_steps: int,
        teacher_model: torch.nn.Module,
        student_model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        pin_memory: bool,
        save_dir: str,
        train_batch_size: int,
        train_set: Dataset,
        valid_batch_size: int,
        valid_set: Dataset,
        collator_fn=None,
        evaluate_on_accuracy: bool = False,
        teacher_finetune_epochs: int = 3,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.001,
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.teacher_finetune_epochs = teacher_finetune_epochs

        self.train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=collator_fn,
        )
        self.valid_loader = DataLoader(
            valid_set,
            batch_size=valid_batch_size,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=collator_fn,
        )

        self.tokenizer = tokenizer
        self.teacher = teacher_model.to(self.device)
        self.student = student_model.to(self.device)

        self.teacher_classifier = nn.Linear(
            self.teacher.config.hidden_size,
            self.student.config.num_labels,
            dtype=torch.float32
        ).to(self.device)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.student.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.student.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

        self.teacher_optimizer = AdamW(
            list(self.teacher.parameters()) + list(self.teacher_classifier.parameters()),
            lr=teacher_learning_rate,
            weight_decay=0.01
        )

        self.kl_divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")
        self.bce_loss_fn = nn.BCEWithLogitsLoss()

        num_training_steps = len(self.train_loader) * epochs
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )

        self.evaluate_on_accuracy = evaluate_on_accuracy
        self.best_valid_score = 0 if evaluate_on_accuracy else float("inf")
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_counter = 0
        self.best_epoch = 0

    def finetune_teacher(self):
        """Fine-tune teacher model và teacher_classifier trên dataset."""
        logger.info("Starting teacher fine-tuning...")
        self.teacher.train()
        self.teacher_classifier.train()

        for epoch in range(self.teacher_finetune_epochs):
            train_loss = AverageMeter()
            with tqdm(total=len(self.train_loader), unit="batches") as tepoch:
                tepoch.set_description(f"Teacher fine-tune epoch {epoch + 1}")
                for data in self.train_loader:
                    input_ids = data["input_ids"].to(self.device)
                    attention_mask = data["attention_mask"].to(self.device)
                    labels = data["labels"].to(self.device)

                    teacher_outputs = self.teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    teacher_hidden = teacher_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
                    teacher_logits = self.teacher_classifier(teacher_hidden)  # [batch_size, num_labels]
                    loss = self.bce_loss_fn(teacher_logits, labels)

                    self.teacher_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.teacher.parameters()) + list(self.teacher_classifier.parameters()),
                        max_norm=1.0
                    )
                    self.teacher_optimizer.step()

                    train_loss.update(loss.item(), input_ids.size(0))
                    current_lr = self.teacher_optimizer.param_groups[0]["lr"]
                    tepoch.set_postfix({"train_loss": train_loss.avg, "lr": current_lr})
                    tepoch.update(1)

        logger.info("Teacher fine-tuning completed.")
        self.teacher.eval()
        self.teacher_classifier.eval()

    def train(self) -> None:
        self.finetune_teacher()
        self.student.train()
        self.teacher.eval()

        for epoch in range(1, self.epochs + 1):
            train_loss = AverageMeter()

            with tqdm(total=len(self.train_loader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")
                for data in self.train_loader:
                    input_ids = data["input_ids"].to(self.device)
                    attention_mask = data["attention_mask"].to(self.device)
                    labels = data["labels"].to(self.device)

                    with torch.no_grad():
                        teacher_outputs = self.teacher(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                        teacher_hidden = teacher_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
                        teacher_logits = self.teacher_classifier(teacher_hidden)  # [batch_size, num_labels]
                        teacher_probs = teacher_logits.softmax(dim=-1) # soft_labels

                    # Student dự đoán
                    student_outputs = self.student(input_ids=input_ids, attention_mask=attention_mask)
                    student_logits = student_outputs.logits
                    student_log_probs = torch.log_softmax(student_logits, dim=-1)

                    kl_loss = self.kl_divergence_loss_fn(student_log_probs, teacher_probs) # Loss với nhãn từ distillation
                    bce_loss = self.bce_loss_fn(student_logits, labels) # Loss với nhãn thật

                    alpha = 0.5
                    loss = alpha * kl_loss + (1 - alpha) * bce_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()

                    train_loss.update(loss.item(), input_ids.size(0))
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    tepoch.set_postfix({"train_loss": train_loss.avg, "lr": current_lr})
                    tepoch.update(1)

            valid_score = self.evaluate(self.valid_loader)
            improved = False

            if self.evaluate_on_accuracy:
                if valid_score > self.best_valid_score + self.early_stopping_threshold:
                    print(f"Validation accuracy improved from {self.best_valid_score:.4f} to {valid_score:.4f}. Saving...")
                    self.best_valid_score = valid_score
                    self.best_epoch = epoch
                    self._save()
                    self.early_stopping_counter = 0
                    improved = True
                    print(f"Saved best model.")
                else:
                    self.early_stopping_counter += 1
                    print(f"No improvement in val accuracy. Counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
            else:
                if valid_score < self.best_valid_score - self.early_stopping_threshold:
                    print(f"Validation loss decreased from {self.best_valid_score:.4f} to {valid_score:.4f}. Saving...")
                    self.best_valid_score = valid_score
                    self.best_epoch = epoch
                    self._save()
                    self.early_stopping_counter = 0
                    improved = True
                    print(f"Saved best model.")
                else:
                    self.early_stopping_counter += 1
                    print(f"No improvement in validation loss. Counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
            
            if improved:
                print(f"Saved best model at epoch {self.best_epoch}.")
            
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {self.early_stopping_patience} epochs without improvement.")
                break

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.student.eval()
        self.teacher.eval()
        eval_loss = AverageMeter()
        all_preds = []
        all_labels = []

        with tqdm(total=len(dataloader), unit="batches") as tepoch:
            tepoch.set_description("validation")
            for data in dataloader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                labels = data["labels"].to(self.device)
                
                teacher_outputs = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_hidden = teacher_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
                teacher_logits = self.teacher_classifier(teacher_hidden)
                teacher_probs = teacher_logits.softmax(dim=-1)
                teacher_preds = (teacher_probs > 0.5).float().cpu().numpy()

                student_outputs = self.student(input_ids=input_ids, attention_mask=attention_mask)
                student_logits = student_outputs.logits
                student_log_probs = torch.log_softmax(student_logits, dim=-1)
                student_probs = torch.softmax(student_logits, dim=-1)
                student_preds = (student_probs > 0.5).float().cpu().numpy()

                kl_loss = self.kl_divergence_loss_fn(student_log_probs, teacher_probs)
                bce_loss = self.bce_loss_fn(student_logits, labels)

                alpha = 0.5
                loss = alpha * kl_loss + (1 - alpha) * bce_loss

                eval_loss.update(loss.item(), input_ids.size(0))

                all_preds.append(student_preds)
                all_labels.append(teacher_preds)

                tepoch.set_postfix({"valid_loss": eval_loss.avg})
                tepoch.update(1)

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        accuracy = np.mean(all_preds == all_labels)
        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        logger.info(f"\n=== Validation Metrics ===")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1-score: {f1 * 100:.2f}%")
        
        return accuracy if self.evaluate_on_accuracy else eval_loss.avg

    def _save(self) -> None:
        self.tokenizer.save_pretrained(self.save_dir)
        self.student.save_pretrained(self.save_dir)
        self.teacher.save_pretrained(f"{self.save_dir}/teacher")
        torch.save(self.teacher_classifier.state_dict(), f"{self.save_dir}/teacher_classifier.pt")
