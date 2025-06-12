from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax

# === 1. Carregar base de dados ===
df = pd.read_csv("dataset_balanceado_70_30.csv")

# === 2. Dividir os dados ===
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['texto'].tolist(), df['sensivel'].tolist(), test_size=0.2, random_state=42
)

# === 3. Preparar os textos ===
def limpar_texto(texto):
    return " ".join(str(texto).split())

train_df = pd.DataFrame({"texto": train_texts, "labels": train_labels})
val_df = pd.DataFrame({"texto": val_texts, "labels": val_labels})

train_df["texto"] = train_df["texto"].apply(limpar_texto)
val_df["texto"] = val_df["texto"].apply(limpar_texto)

# === 4. Tokenização ===
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

# === 5. Carregar modelo ===
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# === 6. Definir métricas ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary')
    return {"accuracy": acc, "f1": f1}

# === 7. Argumentos de treinamento ===
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no"
)

# === 8. Criar Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# === 9. Treinar o modelo ===
trainer.train()

# === 10. Avaliar o modelo ===
eval_results = trainer.evaluate()
print("\U0001F50D Resultados de validação:")
print(f"Acurácia: {eval_results['eval_accuracy']*100:.2f}%")
print(f"F1 Score: {eval_results['eval_f1']*100:.2f}%")

# === 11. Gerar e plotar boxplot das probabilidades preditas ===
def obter_probabilidades_e_labels(model, dataset, batch_size=8):
    model.eval()
    probs = []
    labels = []
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    for batch in dataloader:
        inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()
        batch_probs = softmax(logits, axis=1)[:, 1]  # prob classe 1
        probs.extend(batch_probs)
        labels.extend(batch['labels'].cpu().numpy())
    return np.array(probs), np.array(labels)

probs, true_labels = obter_probabilidades_e_labels(model, val_dataset)

df_probs = pd.DataFrame({
    "Probabilidade Classe Sensível": probs,
    "Label Verdadeiro": ["Sensível" if l == 1 else "Não Sensível" for l in true_labels]
})

plt.figure(figsize=(8,6))
sns.boxplot(x="Label Verdadeiro", y="Probabilidade Classe Sensível", data=df_probs)
plt.title("Distribuição das Probabilidades Preditas pelo Modelo")
plt.show()

# === 12. Salvar o modelo e o tokenizer ===
model.save_pretrained("./modelo_classificador")
tokenizer.save_pretrained("./modelo_classificador")

# === 13. Função para classificar novos textos ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def classificar_texto(texto):
    texto_limpo = limpar_texto(texto)
    inputs = tokenizer(texto_limpo, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Sensível (LGPD)" if prediction == 1 else "Não sensível"

# === 14. Interface simples ===
print("\n=== Classificador de Dados Sensíveis ===")
while True:
    entrada = input("Digite um texto (ou 'sair' para encerrar): ")
    if entrada.lower() == "sair":
        break
    print("\u27A1\uFE0F", classificar_texto(entrada), "\n")
