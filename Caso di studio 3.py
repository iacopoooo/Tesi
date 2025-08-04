import pandas as pd
import numpy as np
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve, auc, precision_score, recall_score, f1_score
import re
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Caricamento file ---
log_file = 'HDFS_100k.log_structured.csv'
label_file = 'anomaly_label.csv'

df_logs = pd.read_csv(log_file)
df_labels = pd.read_csv(label_file)

# --- 2. Parsing dei log ---
log_data = defaultdict(list)
for _, row in df_logs.iterrows():
    content = row["Content"]
    event_id = row["EventId"]
    match = re.search(r'blk_-?\d+', content)
    if match:
        blk_id = match.group(0)
        log_data[blk_id].append(event_id)

# --- 3. Etichette ---
label_dict = dict(zip(df_labels['BlockId'], df_labels['Label'].apply(lambda x: 1 if x == 'Anomaly' else 0)))

# --- 4. Costruzione vocabolario ---
all_events = set(e for seq in log_data.values() for e in seq)
event_vocab = {e: i + 1 for i, e in enumerate(sorted(all_events))}
vocab_size = len(event_vocab) + 1
max_seq_length = 50

# --- 5. Sequenze numeriche e label ---
sequences = []
sequence_labels = []
for blk_id, events in log_data.items():
    if blk_id in label_dict:
        numeric_seq = [event_vocab[e] for e in events]
        sequences.append(numeric_seq)
        sequence_labels.append(label_dict[blk_id])

X = pad_sequences(sequences, maxlen=max_seq_length, padding='post', truncating='post')
y = np.array(sequence_labels)

print(f"Trovate {len(X)} sequenze. Vocabolario di {vocab_size} eventi.")
print(f"Distribuzione classi: Normali={sum(y==0)}, Anomale={sum(y==1)}")

# --- 6. Split train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 7. Modello LSTM bidirezionale ---
model_lstm = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_seq_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_lstm.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

# --- 8. Addestramento ---
history = model_lstm.fit( # Salviamo l'output di fit() nella variabile 'history'
    X_train,
    y_train,
    epochs=15,
    batch_size=64,
    validation_data=(X_test, y_test),
    class_weight={0: 1, 1: 10}
)

# --- 9. Valutazione ---
y_pred_prob = model_lstm.predict(X_test).ravel()
y_pred_class = (y_pred_prob > 0.5).astype(int)

print("\n--- RISULTATI DELLA VALUTAZIONE ---")
print("\nAccuracy:", accuracy_score(y_test, y_pred_class))
print("AUC-ROC Score:", roc_auc_score(y_test, y_pred_prob))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_class))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_class))

# Calcolo metriche per il grafico a barre principale
precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)
accuracy = accuracy_score(y_test, y_pred_class)
roc_auc = roc_auc_score(y_test, y_pred_prob)

# --- 10. Visualizzazione dei Risultati ---

# Grafico 1: Riepilogo delle Metriche Principali
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
values = [accuracy, precision, recall, f1, roc_auc]
plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color='skyblue')
plt.ylim(0, 1.1)
plt.title('Metriche di Valutazione del Modello LSTM')
plt.ylabel('Score')
plt.grid(axis='y')
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

# --- 11. Grafici Aggiuntivi per un'Analisi Approfondita ---

# Grafico 2: Curve di Apprendimento (Loss e Accuracy)
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
accuracy_values = history_dict['accuracy']
val_accuracy_values = history_dict['val_accuracy']
epochs = range(1, len(loss_values) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot della Loss
ax1.plot(epochs, loss_values, 'bo-', label='Training Loss')
ax1.plot(epochs, val_loss_values, 'ro--', label='Validation Loss')
ax1.set_title('Training and Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Plot dell'Accuracy
ax2.plot(epochs, accuracy_values, 'bo-', label='Training Accuracy')
ax2.plot(epochs, val_accuracy_values, 'ro--', label='Validation Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True)

plt.suptitle('Curve di Apprendimento del Modello LSTM')
plt.show()


# Grafico 3: Curva Precision-Recall (PR Curve)
precision_curve_vals, recall_curve_vals, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall_curve_vals, precision_curve_vals)

plt.figure(figsize=(8, 6))
plt.plot(recall_curve_vals, precision_curve_vals, marker='.', label=f'LSTM (AUC-PR = {pr_auc:.2f})')
# Linea di un classificatore casuale (baseline)
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill Baseline')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall')
plt.legend()
plt.grid(True)
plt.show()


# Grafico 4: Distribuzione delle Probabilità Predette
df_results = pd.DataFrame({'Classe Reale': y_test, 'Probabilità Predetta': y_pred_prob})
df_results['Classe Reale'] = df_results['Classe Reale'].map({0: 'Normale', 1: 'Anomalia'})

plt.figure(figsize=(10, 6))
sns.histplot(data=df_results, x='Probabilità Predetta', hue='Classe Reale', kde=True, bins=50, palette={'Normale':'skyblue', 'Anomalia':'red'})
plt.title('Distribuzione delle Probabilità Predette per Classe')
plt.xlabel('Probabilità Predetta di Anomalia')
plt.ylabel('Conteggio Campioni')
plt.grid(True)
plt.show()