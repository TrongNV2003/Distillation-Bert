import json
import time
import numpy as np
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
        true_labels = []
        pred_labels = []
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

                pred_labels.extend(student_preds)
                true_labels.extend(teacher_preds)

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

        self._print_metrics(true_labels, pred_labels, "micro")
        self._print_metrics(true_labels, pred_labels, "macro")
        self._print_metrics(true_labels, pred_labels, "weighted")
        self._calculate_accuracy(results)
        self._calculate_latency(latencies)
        print(f"num samples: {num_samples}")

    def _map_labels(self, label_indices: list, labels_mapping: dict) -> list:
        return [labels_mapping[idx] for idx, val in enumerate(label_indices) if val == 1.0]

    def _print_metrics(self, true_labels, pred_labels, average_type):
        precision = precision_score(true_labels, pred_labels, average=average_type, zero_division=0)
        recall = recall_score(true_labels, pred_labels, average=average_type, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, average=average_type, zero_division=0)
        print(f"\nMetrics ({average_type}):")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1 Score: {f1 * 100:.2f}%")

    def _calculate_accuracy(self, data):
        correct = 0
        correct_one = 0
        total = len(data)
        for item in data:
            true_set = set(item["true_labels"])
            pred_set = set(item["predicted_labels"])
            if true_set == pred_set:
                correct += 1
            if true_set & pred_set:
                correct_one += 1
        accuracy = correct / total if total > 0 else 0
        accuracy_one = correct_one / total if total > 0 else 0
        print(f"\nAccuracy (Match one): {accuracy_one * 100:.2f}%")
        print(f"Accuracy (Match all): {accuracy * 100:.2f}%")

    def _calculate_latency(self, latencies: list) -> None:
        p99_latency = np.percentile(latencies, 99)
        print(f"\nP99 Latency: {p99_latency * 1000:.2f} ms")