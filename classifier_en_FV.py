import os
FULL_DETERMINISM = True
if FULL_DETERMINISM:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import set_seed
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
import re
import json
import platform
import transformers
import sklearn


REMOVE_DATES = True

SUFFIX = "no_dates" if REMOVE_DATES else "with_dates"
print(f"\n>>> Scenario: {'DATES REMOVED' if REMOVE_DATES else 'DATES KEPT (baseline)'}  (suffix: {SUFFIX})\n")


set_seed(42)


if FULL_DETERMINISM:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    torch.use_deterministic_algorithms(True)
    print(">>> FULL_DETERMINISM enabled (cuDNN deterministic, deterministic algorithms, CUBLAS workspace set)")


ENV_INFO = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "numpy": np.__version__,
    "sklearn": sklearn.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "cudnn_version": (torch.backends.cudnn.version() if torch.cuda.is_available() else None),
    "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
    "seed": 42,
    "full_determinism": FULL_DETERMINISM,
    "remove_dates": REMOVE_DATES,
}
print("\n=== Environment ===")
for _k, _v in ENV_INFO.items():
    print(f"  {_k}: {_v}")
print()


_MONTHS = (r'jan(eiro)?|fev(ereiro)?|mar(ço|co)?|abr(il)?|mai(o)?|jun(ho)?|'
           r'jul(ho)?|ago(sto)?|set(embro)?|out(ubro)?|nov(embro)?|dez(embro)?')
_DATE_PATTERNS = [
    re.compile(r'\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b'),
    re.compile(r'\b(19|20)\d{2}\b'),
    re.compile(r'\b\d{1,2}\s*(de\s*)?(%s)\b' % _MONTHS, re.I),
    re.compile(r'\b(%s)\b' % _MONTHS, re.I),
]

def strip_dates(text):
    text = str(text)
    for p in _DATE_PATTERNS:
        text = p.sub(' ', text)
    return text


df = pd.read_csv("dataset_balanceado_70_30.csv")


train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["texto"].tolist(), df["sensivel"].tolist(), test_size=0.2, random_state=42
)


def clean_text(text):

    if REMOVE_DATES:
        text = strip_dates(text)
    return " ".join(str(text).split())

train_df = pd.DataFrame({"texto": train_texts, "labels": train_labels})
val_df = pd.DataFrame({"texto": val_texts, "labels": val_labels})

train_df["texto"] = train_df["texto"].apply(clean_text)
val_df["texto"] = val_df["texto"].apply(clean_text)


model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["texto"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


set_seed(42)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="binary")
    return {"accuracy": acc, "f1": f1}


training_args = TrainingArguments(
    output_dir=f"./results_{SUFFIX}",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir=f"./logs_{SUFFIX}",
    logging_steps=10,
    save_strategy="no",
    seed=42,
    full_determinism=FULL_DETERMINISM
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)


trainer.train()


eval_results = trainer.evaluate()
print(f"\nValidation results [{SUFFIX}]:")
print(f"Accuracy: {eval_results['eval_accuracy']*100:.2f}%")
print(f"F1 score: {eval_results['eval_f1']*100:.2f}%")


def get_probs_and_labels(model, dataset, batch_size=8):
    model.eval()
    probs = []
    labels = []
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    for batch in dataloader:
        inputs = {k: v.to(model.device) for k, v in batch.items() if k != "labels"}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()
        batch_probs = softmax(logits, axis=1)[:, 1]
        probs.extend(batch_probs)
        labels.extend(batch["labels"].cpu().numpy())
    return np.array(probs), np.array(labels)

probs, true_labels = get_probs_and_labels(model, val_dataset)


pred_labels = (probs >= 0.5).astype(int)
print(f"\n=== Confusion matrix [{SUFFIX}] [rows=true 0,1][cols=pred 0,1] ===")
print(confusion_matrix(true_labels, pred_labels, labels=[0, 1]))
print(f"\n=== Classification report [{SUFFIX}] (per-class P/R/F1 + macro) ===")
print(classification_report(
    true_labels, pred_labels,
    target_names=["Not sensitive (0)", "Sensitive (1)"], digits=4
))
pd.DataFrame({"prob_sensitive": probs, "true_label": true_labels}).to_csv(
    f"val_predictions_{SUFFIX}.csv", index=False
)
print(f"Saved: val_predictions_{SUFFIX}.csv")


cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
report_dict = classification_report(
    true_labels, pred_labels,
    target_names=["Not sensitive (0)", "Sensitive (1)"],
    output_dict=True, digits=4
)
metrics_out = {
    "scenario": SUFFIX,
    "accuracy": float(accuracy_score(true_labels, pred_labels)),
    "f1_sensitive_binary": float(f1_score(true_labels, pred_labels, average="binary")),
    "macro_f1": float(f1_score(true_labels, pred_labels, average="macro")),
    "confusion_matrix_rows_true_cols_pred": cm.tolist(),
    "per_class_report": report_dict,
    "config": {
        "model_name": model_name,
        "num_train_epochs": 5,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "max_length": 128,
        "seed": 42,
        "remove_dates": REMOVE_DATES,
        "full_determinism": FULL_DETERMINISM,
    },
    "environment": ENV_INFO,
}
with open(f"metrics_{SUFFIX}.json", "w", encoding="utf-8") as fh:
    json.dump(metrics_out, fh, indent=2, ensure_ascii=False)
print(f"Saved: metrics_{SUFFIX}.json")


df_probs = pd.DataFrame({
    "Probability of Sensitive Class": probs,
    "True Label": ["Sensitive" if l == 1 else "Not Sensitive" for l in true_labels]
})

plt.figure(figsize=(8, 6))
sns.boxplot(
    x="True Label",
    y="Probability of Sensitive Class",
    data=df_probs,
    order=["Sensitive", "Not Sensitive"]
)
plt.title(f"Distribution of Predicted Probabilities ({'dates removed' if REMOVE_DATES else 'dates kept'})")
plt.ylim(-0.02, 1.02)
plt.tight_layout()
plt.savefig(f"figure1_boxplot_{SUFFIX}.png", dpi=300, bbox_inches="tight")
print(f"Saved: figure1_boxplot_{SUFFIX}.png")
plt.show()


model.save_pretrained(f"./modelo_classificador_{SUFFIX}")
tokenizer.save_pretrained(f"./modelo_classificador_{SUFFIX}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def classify_text(text):
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Sensitive (LGPD)" if prediction == 1 else "Not sensitive"


RUN_INTERACTIVE = True
if RUN_INTERACTIVE:
    print("\n=== Sensitive Data Classifier ===")
    while True:
        entry = input("Enter a text (or 'exit' to quit): ")
        if entry.lower() == "exit":
            break
        print("->", classify_text(entry), "\n")
