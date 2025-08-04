import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# --- 1. Preparazione dei Dati ---
# Assicurati che la cartella 'malimg_dataset' sia nella stessa directory dello script
data_dir = 'dataset_9010/dataset_9010/malimg_dataset/train'

if not os.path.exists(data_dir):
    print(f"Errore: La cartella '{data_dir}' non è stata trovata.")
    print("Assicurati di aver scaricato e decompresso il dataset Malimg da Kaggle.")
    exit()

# Dimensioni delle immagini e batch size
img_width, img_height = 64, 64
batch_size = 32

# Usiamo ImageDataGenerator per caricare le immagini dalle cartelle
# e dividerle in training (80%) e validation (20%) set
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalizza i pixel tra 0 e 1
    validation_split=0.2 # 20% dei dati per la validazione
)

# Generatore per il training set
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Generatore per il validation set
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# --- 2. Costruzione del Modello CNN ---
print("\nCostruzione del modello CNN...")

model_cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model_cnn.summary()

# --- 3. Compilazione e Addestramento del Modello ---
model_cnn.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nInizio addestramento della CNN...")
history = model_cnn.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)
print("Addestramento completato.")

# --- 4. Valutazione del Modello ---
print("\nValutazione finale sul validation set...")
Y_val = validation_generator.classes
y_pred_prob = model_cnn.predict(validation_generator)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\nRapporto di Classificazione:")
print(classification_report(Y_val, y_pred, target_names=validation_generator.class_indices.keys()))

# --- 5. Visualizzazione dei Risultati Principali ---

# Grafico 1: Matrice di Confusione con Conteggi Assoluti
print("\nMatrice di Confusione:")
cm = confusion_matrix(Y_val, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(cm, annot=False, cmap='viridis',
            xticklabels=validation_generator.class_indices.keys(),
            yticklabels=validation_generator.class_indices.keys())
plt.title('Matrice di Confusione - CNN su Malimg')
plt.ylabel('Classe Reale')
plt.xlabel('Classe Predetta')
plt.show()

# Grafico 2: Metriche per Classe (Precision, Recall, F1-Score)
precision, recall, f1, _ = precision_recall_fscore_support(Y_val, y_pred, zero_division=0)
class_names = list(validation_generator.class_indices.keys())
metrics_df = pd.DataFrame({
    'Classe': class_names,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
})
metrics_df.set_index('Classe')[['Precision', 'Recall', 'F1-Score']].plot(kind='bar', figsize=(18, 6))
plt.title('Metriche per Classe - CNN su Malimg')
plt.ylabel('Punteggio')
plt.ylim(0, 1.1)
plt.grid(axis='y')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# --- 6. Grafici Aggiuntivi per un'Analisi Approfondita ---

print("\nGenerazione grafici di analisi avanzata...")

# Grafico 3: Curve di Apprendimento
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
accuracy_values = history_dict['accuracy']
val_accuracy_values = history_dict['val_accuracy']
epochs = range(1, len(loss_values) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
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
plt.suptitle('Curve di Apprendimento del Modello CNN')
plt.show()

# Grafico 4: Matrice di Confusione Normalizzata (per visualizzare il Recall)
# Normalizziamo per riga per vedere le percentuali
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(14, 12))
sns.heatmap(cm_normalized, annot=False, cmap='cividis',
            xticklabels=validation_generator.class_indices.keys(),
            yticklabels=validation_generator.class_indices.keys(),
            vmin=0, vmax=1) # Scala di colori da 0 a 1
plt.title('Matrice di Confusione Normalizzata (Recall per Classe)')
plt.ylabel('Classe Reale')
plt.xlabel('Classe Predetta')
plt.show()

# Grafico 5: Le 10 Misclassificazioni più Comuni
# Creiamo una copia della matrice per non modificare l'originale
cm_errors = cm.copy()
np.fill_diagonal(cm_errors, 0)
flat_cm = cm_errors.flatten()
top_indices = np.argsort(flat_cm)[-10:]
top_errors = flat_cm[top_indices]
row_indices, col_indices = np.unravel_index(top_indices, cm_errors.shape)

error_labels = []
class_names_list = list(validation_generator.class_indices.keys())
for r, c in zip(row_indices, col_indices):
    true_class = class_names_list[r]
    pred_class = class_names_list[c]
    error_labels.append(f"Reale: {true_class}\nPredetta: {pred_class}")

plt.figure(figsize=(12, 8))
sns.barplot(x=top_errors, y=error_labels, palette='Reds_r', orient='h')
plt.title('Le 10 Misclassificazioni più Frequenti')
plt.xlabel('Numero di Campioni Errati')
plt.ylabel('Tipo di Errore')
plt.tight_layout()
plt.show()