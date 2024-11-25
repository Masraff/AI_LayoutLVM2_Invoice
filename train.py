from transformers import LayoutLMv2ForTokenClassification, LayoutLMv2Processor, TrainingArguments, Trainer, \
    DataCollatorForTokenClassification, EarlyStoppingCallback
import torch
import os
import re
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
print(torch.__version__)
print(torch.cuda.is_available())

device = torch.device("cuda")
print(device)
# Directories
train_data_dir = r'train_data'
test_data_dir = r'test_data'
output_model_dir = r'saved_modelv2'
os.makedirs(output_model_dir, exist_ok=True)

# Load LayoutLMv2 processor
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", apply_ocr=False)
model = LayoutLMv2ForTokenClassification.from_pretrained("microsoft/layoutlmv2-base-uncased", num_labels=2)
model.to(device)

# Dataset class
class InvoiceDataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = torch.load(self.data_files[idx], weights_only=True)
        # Ensure image and bbox are correctly structured
        data['bbox'] = torch.clamp(data.get('bbox', torch.zeros((512, 4))), min=0, max=1000).to(torch.int64)
        data['input_ids'] = data.get('input_ids', torch.zeros(512, dtype=torch.int64))
        data['attention_mask'] = data.get('attention_mask', torch.ones(512, dtype=torch.int64))
        data['labels'] = data.get('labels', torch.full((512,), -100, dtype=torch.int64))
        data['image'] = data.get('image', torch.zeros((3, 224, 224), dtype=torch.float))  # Ensure image tensor is provided
        return {key: value.squeeze(0) for key, value in data.items()}

# Load train and test datasets
train_dataset = InvoiceDataset(train_data_dir)
test_dataset = InvoiceDataset(test_data_dir)

# Gather all labels in the training dataset
all_labels = []
for i in range(len(train_dataset)):
    data = train_dataset[i]
    all_labels.extend(data['labels'].view(-1).cpu().numpy())

# Compute balanced class weights

# Recompute class weights without 'balanced' parameter
class_counts = np.bincount(all_labels)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()  # Normalize
class_weights = torch.tensor(class_weights, dtype=torch.float).to(model.device)


# Focal loss
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, logits, labels):
        ce_loss = self.ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss

focal_loss_function = FocalLoss(weight=class_weights)

# Trainer with custom loss and token importance
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, *args, **kwargs):
        labels = inputs.pop("labels")

        # Mask words without digits
        labels_np = labels.cpu().numpy()
        input_ids_np = inputs["input_ids"].cpu().numpy()

        token_weights = np.ones_like(labels_np, dtype=np.float32)
        # Adjust token weights
        for i in range(len(input_ids_np)):
            for j in range(len(input_ids_np[i])):
                token = processor.tokenizer.decode([input_ids_np[i][j]], skip_special_tokens=True)
                if not re.search(r'\d', token):
                    token_weights[i][j] = 0.5  # Adjusted weight

        token_weights = torch.tensor(token_weights, dtype=torch.float, device=model.device)

        labels = torch.tensor(labels_np, dtype=torch.long, device=model.device)

        outputs = model(**inputs)
        logits = outputs.logits

        # Apply Focal Loss and token weighting
        loss = focal_loss_function(logits, labels)
        loss = loss * token_weights.view(-1)  # Adjust loss with token weights
        loss = loss.mean()

        return (loss, outputs) if kwargs.get("return_outputs", False) else loss


# Custom data collator
class CustomDataCollator(DataCollatorForTokenClassification):
    def __call__(self, features):
        for feature in features:
            feature['input_ids'] = feature.get('input_ids', torch.zeros(512, dtype=torch.int64))
            feature['attention_mask'] = feature.get('attention_mask', torch.ones(512, dtype=torch.int64))
            feature['bbox'] = feature.get('bbox', torch.zeros((512, 4), dtype=torch.int64))
            feature['labels'] = feature.get('labels', torch.full((512,), -100, dtype=torch.int64))
            feature['image'] = feature.get('image', torch.zeros((3, 224, 224), dtype=torch.float))
        return super().__call__(features)

data_collator = CustomDataCollator(tokenizer=processor.tokenizer, padding=True)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_model_dir,
    eval_strategy="epoch",  # Evaluation is performed at the end of each epoch
    save_strategy="epoch",  # Save the model at the end of each epoch to match eval_strategy
    learning_rate=1e-5,  # Reduced learning rate
    per_device_train_batch_size=8,  # Increased batch size
    per_device_eval_batch_size=8,
    num_train_epochs=15,  # Increased epochs
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True  # Automatically load the best model at the end
)


# Compute metrics
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=2).flatten()
    labels = p.label_ids.flatten()

    preds_filtered = [pred for pred, label in zip(preds, labels) if label != -100]
    labels_filtered = [label for label in labels if label != -100]

    precision, recall, f1, _ = precision_recall_fscore_support(labels_filtered, preds_filtered, average='binary', pos_label=1)
    accuracy = accuracy_score(labels_filtered, preds_filtered)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Trainer
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Early stopping
)

# Train and evaluate the model
trainer.train()
trainer.evaluate()

# Save the model
model.save_pretrained(output_model_dir)
print("Model training and evaluation completed. Model saved to", output_model_dir)
