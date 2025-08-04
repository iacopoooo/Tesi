import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Caricamento e Campionamento dei Dati ---
# Nota: il dataset completo è molto grande. Per scopi didattici,
# carichiamo un campione casuale del 10% delle righe.
# Se il tuo PC ha poca RAM, puoi ridurre ulteriormente questa percentuale.
try:
    # Carica un campione del dataset per non esaurire la memoria
    df = pd.read_csv('cicids2017.csv', iterator=True, chunksize=10000)
    df = pd.concat([chunk.sample(frac=0.1, random_state=42) for chunk in df])
    print("Dataset caricato con successo.")
except FileNotFoundError:
    print("Errore: 'cicids2017.csv' non trovato. Assicurati che il file sia nella stessa directory.")
    exit()

# Rinomina le colonne per rimuovere spazi e caratteri problematici
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
print(f"Shape del dataset campionato: {df.shape}")
print("\nPrime 5 righe del dataset:")
print(df.head())


# --- 2. Pre-processing e Pulizia dei Dati ---
print("\nInizio pre-processing...")

# Rimuovi valori infiniti e NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Definisci le feature (X) e il target (y)
# Supponiamo che l'ultima colonna sia 'label'
X = df.drop('label', axis=1)
y = df['label']

print(f"Distribuzione delle classi nel target (y):\n{y.value_counts()}")

# Normalizzazione delle feature numeriche
# È fondamentale per molti algoritmi di ML
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# --- 3. Suddivisione del Dataset in Training e Test ---
# 80% per l'addestramento, 20% per la valutazione
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y # stratify è utile per dati sbilanciati
)
print("\nDataset suddiviso in training e test set.")
print(f"Shape di X_train: {X_train.shape}, Shape di X_test: {X_test.shape}")


# --- 4. Addestramento del Modello (Random Forest) ---
print("\nInizio addestramento del modello Random Forest...")
# n_estimators è il numero di alberi nella foresta.
# n_jobs=-1 usa tutti i core della CPU per velocizzare.
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

rf_classifier.fit(X_train, y_train)
print("Addestramento completato.")


# --- 5. Valutazione delle Performance ---
print("\nInizio valutazione del modello sul test set...")
y_pred = rf_classifier.predict(X_test)

# a) Accuratezza generale
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuratezza Generale: {accuracy * 100:.2f}%")

# b) Rapporto di Classificazione (Precision, Recall, F1-Score)
print("\nRapporto di Classificazione:")
# zero_division=0 evita warning se una classe non ha predizioni
print(classification_report(y_test, y_pred, zero_division=0))

# c) Matrice di Confusione
print("\nMatrice di Confusione:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot della matrice di confusione per una migliore visualizzazione
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=rf_classifier.classes_, yticklabels=rf_classifier.classes_)
plt.title('Matrice di Confusione - Random Forest')
plt.ylabel('Classe Reale')
plt.xlabel('Classe Predetta')
plt.show()

# d) Importanza delle Feature (opzionale ma molto utile)
print("\nLe 10 feature più importanti secondo il modello:")
feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importances.head(10))

# --- Grafico Aggiuntivo: Importanza delle Feature ---
plt.figure(figsize=(10, 8))
top_20_features = feature_importances.head(20)
sns.barplot(x=top_20_features, y=top_20_features.index, palette='viridis')
plt.title('Le 20 Feature più Importanti - Random Forest')
plt.xlabel('Importanza Relativa')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
# ... (tutto il codice fino alla fine della valutazione della Random Forest) ...

# -----------------------------------------------------------------
# --- 6. Addestramento e Valutazione di un Modello Deep Learning (MLP) ---
# -----------------------------------------------------------------

from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings

print("\n\n" + "="*50)
print("Inizio addestramento e valutazione del modello MLP (Deep Learning)...")
print("="*50)

# Creiamo il classificatore MLP.
# hidden_layer_sizes: (50, 25) significa due strati nascosti, uno con 50 neuroni, l'altro con 25.
# max_iter: numero massimo di epoche di addestramento.
# alpha: termine di regolarizzazione per evitare overfitting.
# early_stopping: ferma l'addestramento se il modello non migliora, per evitare overfitting.
mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(50, 25),
    max_iter=300,
    alpha=1e-4,
    solver='adam',
    verbose=10, # Stampa il progresso dell'addestramento
    random_state=42,
    tol=1e-4,
    learning_rate_init=.001,
    early_stopping=True, # Molto utile!
    validation_fraction=0.1 # Usa il 10% dei dati di training per la validazione interna
)

# Ignoriamo i warning di convergenza per mantenere l'output pulito
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp_classifier.fit(X_train, y_train)

print("Addestramento MLP completato.")

# --- Valutazione delle Performance dell'MLP ---
print("\nInizio valutazione del modello MLP sul test set...")
y_pred_mlp = mlp_classifier.predict(X_test)

# a) Accuratezza generale MLP
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"\nAccuratezza Generale (MLP): {accuracy_mlp * 100:.2f}%")

# b) Rapporto di Classificazione MLP
print("\nRapporto di Classificazione (MLP):")
print(classification_report(y_test, y_pred_mlp, zero_division=0))

# c) Matrice di Confusione MLP
print("\nMatrice di Confusione (MLP):")
cm_mlp = confusion_matrix(y_test, y_pred_mlp)
print(cm_mlp)

# Plot della matrice di confusione dell'MLP
plt.figure(figsize=(12, 8))
sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Greens', xticklabels=mlp_classifier.classes_, yticklabels=mlp_classifier.classes_)
plt.title('Matrice di Confusione - MLP (Deep Learning)')
plt.ylabel('Classe Reale')
plt.xlabel('Classe Predetta')
plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# Supponiamo di avere:
# y_test: target reale
# y_pred: predizioni Random Forest
# y_pred_mlp: predizioni MLP
# Le etichette devono essere stringhe per i tick chiari
class_labels = sorted(np.unique(y_test).astype(str))

# Calcolo delle metriche
precision_rf, recall_rf, f1_rf, _ = precision_recall_fscore_support(y_test, y_pred, labels=np.unique(y_test), zero_division=0)
precision_mlp, recall_mlp, f1_mlp, _ = precision_recall_fscore_support(y_test, y_pred_mlp, labels=np.unique(y_test), zero_division=0)

# --- 3 Grafici a barre affiancati ---
x = np.arange(len(class_labels))
width = 0.35

fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Precision
axs[0].bar(x - width/2, precision_rf, width, label='Random Forest', color='orange')
axs[0].bar(x + width/2, precision_mlp, width, label='MLP', color='orangered')
axs[0].set_title('Precision Comparison')
axs[0].set_xticks(x)
axs[0].set_xticklabels(class_labels, rotation=45)

# Recall
axs[1].bar(x - width/2, recall_rf, width, label='Random Forest', color='orange')
axs[1].bar(x + width/2, recall_mlp, width, label='MLP', color='orangered')
axs[1].set_title('Recall Comparison')
axs[1].set_xticks(x)
axs[1].set_xticklabels(class_labels, rotation=45)

# F1-score
axs[2].bar(x - width/2, f1_rf, width, label='Random Forest', color='orange')
axs[2].bar(x + width/2, f1_mlp, width, label='MLP', color='orangered')
axs[2].set_title('F1-score Comparison')
axs[2].set_xticks(x)
axs[2].set_xticklabels(class_labels, rotation=45)

# Etichette comuni
for ax in axs:
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y')
    ax.set_xlabel('Classi')
axs[0].set_ylabel('Score')
axs[2].legend(loc='upper right')

plt.tight_layout()
plt.show()

