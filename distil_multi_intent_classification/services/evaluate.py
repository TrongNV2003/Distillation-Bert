import json
import time
import numpy as np
from loguru import logger
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Tester:
    def __init__(
        self,
        model: torch.nn.Module,
        teacher_model: nn.Module,
        teacher_classifier: nn.Module,
        test_loader: DataLoader,
        output_file: str = None,
    ) -> None:
        self.test_loader = test_loader
        self.output_file = output_file

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.teacher = teacher_model.to(self.device)
        self.teacher.eval()
        self.teacher_classifier = teacher_classifier.to(self.device)
        self.teacher_classifier.eval()

        self.kl_divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")
        self.bce_loss_fn = nn.BCEWithLogitsLoss()

    def evaluate(self):
        self.model.eval()
        latencies = []
        all_labels = []
        all_preds = []
        total_loss = 0
        results = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                batch_start_time = time.time()

                teacher_outputs = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                teacher_hidden = teacher_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
                teacher_logits = self.teacher_classifier(teacher_hidden)
                teacher_probs = teacher_logits.softmax(dim=-1)
                teacher_preds = (teacher_probs > 0.5).float().cpu().numpy()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                student_logits = outputs.logits
                student_log_probs = torch.log_softmax(student_logits, dim=-1)
                student_probs = student_logits.softmax(dim=-1)
                student_preds = (student_probs > 0.5).float().cpu().numpy()

                kl_loss = self.kl_divergence_loss_fn(student_log_probs, teacher_probs)
                bce_loss = self.bce_loss_fn(student_logits, labels)

                alpha = 0.5
                loss = alpha * kl_loss + (1 - alpha) * bce_loss

                total_loss += loss.item()

                batch_end_time = time.time()
                latency = batch_end_time - batch_start_time
                latency_per_sample = latency / input_ids.size(0)
                latencies.append(latency_per_sample)

                all_preds.extend(student_preds)
                all_labels.extend(teacher_preds)

                for i in range(len(student_preds)):
                    true_label_names = self._map_labels(teacher_preds[i], self.model.config.id2label)
                    predicted_label_names = self._map_labels(student_preds[i], self.model.config.id2label)
                    results.append({
                        "true_labels": true_label_names,
                        "predicted_labels": predicted_label_names,
                        "latency": float(latency),
                    })
                    
        num_samples = len(results)
        avg_loss = total_loss / len(self.test_loader)
        print(f"Average Test Loss: {avg_loss:.4f}")
        
        if self.output_file:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"Results saved to {self.output_file}")

        self.score(all_labels, all_preds, results)
        self.calculate_latency(latencies)

        print(f"num samples: {num_samples}")

    def _map_labels(self, label_indices: list, labels_mapping: dict) -> list:
        return [labels_mapping[idx] for idx, val in enumerate(label_indices) if val == 1.0]


    def score(self, label: list, predict: list, output: list) -> None:
        precision = precision_score(label, predict, average="weighted", zero_division=0)
        recall = recall_score(label, predict, average="weighted", zero_division=0)
        f1 = f1_score(label, predict, average="weighted", zero_division=0)
        accuracy = self._accuracy(output)

        logger.info(f"Accuracy: {accuracy * 100:.2f}")
        logger.info(f"Precision: {precision * 100:.2f}")
        logger.info(f"Recall: {recall * 100:.2f}")
        logger.info(f"F1 score: {f1 * 100:.2f}")


    def _accuracy(self, output_data: list) -> float:
        """
        Calculate accuracy for multi-label predictions where a sample is correct
        if at least one predicted label matches the true labels.
        """
        correct = 0
        total = len(output_data)

        for sample in output_data:
            true_labels = set(sample["true_labels"])
            predicted_labels = set(sample["predicted_labels"])
            
            if true_labels & predicted_labels:  # Giao của true_labels và predicted_labels không rỗng
                correct += 1

        return correct / total if total > 0 else 0.0

    def calculate_latency(self, latencies: list) -> None:
        p99_latency = np.percentile(latencies, 99)
        print(f"P99 Latency: {p99_latency * 1000:.2f} ms")