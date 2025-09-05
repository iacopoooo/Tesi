# 📘 Progetti di Machine Learning e Deep Learning per la Sicurezza Informatica

Questo repository contiene **tre casi di studio** su dataset differenti, con l’obiettivo di applicare tecniche di Machine Learning e Deep Learning al rilevamento di minacce e anomalie.

---

#=== CASO DI STUDIO 1 ===#

# Confronto Random Forest vs MLP sul dataset CICIDS2017

Questo progetto confronta due modelli di Machine Learning:
- **Random Forest (RF)**
- **Multi-Layer Perceptron (MLP)**

per il rilevamento di attacchi **Denial of Service (DoS)** all’interno del dataset **CICIDS2017**.

I grafici e i risultati includono:
- Matrici di confusione
- Importanza delle feature (RF)
- Metriche di classificazione (accuratezza, precisione, recall, F1-score)
- Confronto finale tra RF e MLP per ogni classe

---

## Dataset

Il dataset usato è **CICIDS2017**, disponibile pubblicamente:

🔗 [CICIDS2017 su Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)

Scaricare il file CSV corrispondente (es. `cicids2017.csv`) e posizionarlo nella cartella del progetto oppure in Google Colab/Drive.

⚠️ Attenzione: il file è grande (~214 MB).  
Se usi **Colab**, caricalo su Google Drive e montalo nel notebook.

---

## Requisiti

Installare le dipendenze principali:

```bash
pip install -r requirements.txt
```

---

#=== CASO DI STUDIO 2 ===#

# Classificazione Malware con CNN (Malimg Dataset)

Questo progetto implementa una **Rete Neurale Convoluzionale (CNN)** per classificare immagini di malware tratte dal dataset **Malimg**.  
Il dataset contiene rappresentazioni visive di famiglie di malware e viene usato come benchmark per la classificazione automatica.

---

## Dataset

Il dataset utilizzato è **Malimg**, disponibile su Kaggle:

🔗 [Malimg Dataset su Kaggle](https://www.kaggle.com/datasets/)

1. Scarica il dataset da Kaggle.  
2. Estrai la cartella `malimg_dataset` nella directory del progetto, oppure nella struttura:

```
dataset_9010/
└── dataset_9010/
    └── malimg_dataset/
        └── train/
            ├── class_1/
            ├── class_2/
            ├── ...
```

Se il percorso non corrisponde, aggiorna la variabile `data_dir` nello script Python.

---

## Requisiti

Installa le dipendenze principali:

```bash
pip install -r requirements.txt
```

---

#=== CASO DI STUDIO 3 ===#

# Rilevamento Anomalie su Log HDFS con BiLSTM

Questo progetto addestra un modello **BiLSTM** (Keras/TensorFlow) per classificare sequenze di eventi dai log **HDFS** come **Normale** o **Anomalo**.  
Il pipeline include parsing dei log, costruzione del vocabolario di EventId, creazione delle sequenze, training e valutazione (Accuracy, Precision, Recall, F1, ROC-AUC) con grafici.

---

## Dataset

Servono due file (in CSV):

- `HDFS_100k.log_structured.csv` → log strutturati con almeno le colonne:
  - `Content` (riga di log)
  - `EventId` (ID dell’evento estratto/parsato)
- `anomaly_label.csv` → etichette per block:
  - `BlockId` (es. `blk_-12345`)
  - `Label` (`Anomaly` oppure `Normal`)

> **Nota:** lo script estrae il `BlockId` da `Content` tramite regex `r'blk_-?\d+'`.  
> Assicurati che il formato dei log sia coerente (es. dataset HDFS Log pubblici).

Struttura tipica del progetto:
```
├── HDFS_100k.log_structured.csv
├── anomaly_label.csv
├── main.py                # lo script con il modello BiLSTM
├── requirements.txt
└── README.md
```

---

## Requisiti

Installa le dipendenze:

```bash
pip install -r requirements.txt
```

---

# 📜 Licenza

Questo progetto è distribuito sotto licenza **MIT**.  
Puoi usarlo, modificarlo e condividerlo liberamente.

---

# ✨ Autore

Progetti sviluppati da **[Il Tuo Nome]**, a scopo di ricerca e sperimentazione sull’uso di ML/DL per la sicurezza informatica.
