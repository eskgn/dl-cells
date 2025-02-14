import optuna
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold

pd.set_option('display.max_columns', None)

path = "Cells.csv"
df = pd.read_csv(path,sep=';')

#Stockage d'un fichier qui répertorie les paramètres utilisés (pour optimisation)
study_name = 'study_cells'  #nom du fichier
storage_name = 'sqlite:///cells_study.db'
try:
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    print("Étude chargée.")
except KeyError:
    study = optuna.create_study(study_name=study_name, direction='maximize', storage=storage_name)
    print("Nouvelle étude créée.")

#Visualisation des données
# print(df.head())
# print(df.dtypes) 
# print(df['class'].unique())


#Nettoyage
df = df.iloc[:, 2:]
cible = 'class'
df_quant = df.drop(columns=cible)
for col in df_quant:
    df_quant[col] = df_quant[col].astype(str).str.replace(',', '.').astype(float)

df[cible] = df[cible].map({'PS': 0, 'WS': 1})

print(df_quant.dtypes)

#Séparation des données
X = df_quant
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

from collections import Counter
print(Counter(y)) #avant smote

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train, y_train = smote.fit_resample(X, y)

print(Counter(y_train)) #après smote

#Preprocessing

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Conversion en tenseurs PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  #Pour correspondre à la sortie
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

#Création du DataLoader et des batch
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)


#Définition du modèle, choix de 3 couches de neurones
def create_model(input_size, hidden1, hidden2, dropout):
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden1)
            self.fc2 = nn.Linear(hidden1, hidden2)
            self.fc3 = nn.Linear(hidden2, 1) #(1 neurone pour la sortie car classification binaire)
            self.dropout = nn.Dropout(dropout) #Eviter le surapprentissage
            self.relu = nn.LeakyReLU() #Fonction d'activation efficace avec bonne convergence et simplification, meilleure performance que ReLU
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.fc3(x) #Pas de ReLU, utilisation de sigmoid pour sortie avec une probabilité entre 0 et 1 
            return torch.sigmoid(x)
    
    return NeuralNetwork()

#Fonction d'optimisation avec Optuna avec sélection des bornes et valeurs des hyperparamètres
def objective(trial):
    hidden1 = trial.suggest_int('hidden1', 256, 1500)
    hidden2 = trial.suggest_int('hidden2', 64, 1024)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-7, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

    #Intégration de fold pour éviter le surapprentissage
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    
    for train_idx, val_idx in kf.split(X_train, y_train):
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        model = create_model(X_train.shape[1], hidden1, hidden2, dropout)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(50):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
        
        model.eval()
        y_true, y_scores = [], []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                y_pred_val = model(X_val)
                y_pred_val = torch.sigmoid(y_pred_val).cpu().numpy()
                y_true.extend(y_val.cpu().numpy())
                y_scores.extend(y_pred_val)
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_scores.append(auc(fpr, tpr))
    
    return np.mean(auc_scores)

#Exécution d'Optuna
study.optimize(objective, n_trials=1) #désigne le nombre de paramètre testé pour la boucle

#Meilleurs hyperparamètres
best_params = study.best_params
print("Meilleurs hyperparamètres :", best_params)

#Entraînement
best_model = create_model(X_train.shape[1], best_params['hidden1'], best_params['hidden2'], best_params['dropout'])
criterion = nn.BCELoss()
optimizer = optim.Adam(best_model.parameters(), lr=best_params['lr'])  #choix d'Adam (version plus efficace que le SGD)
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
train_losses = []
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.1) #Permet de diminuer le lr lorsque l'epoch stagne

for epoch in range(75):
    best_model.train()
    epoch_loss = 0  #Initialisation de la loss pour l'époque
    for X_batch, y_batch in train_loader: #Itération sur les batchs
        optimizer.zero_grad()
        y_pred = best_model(X_batch)
        loss = criterion(y_pred, y_batch) #Calcul du loss
        loss.backward()
        optimizer.step() #Mise à jour des poids
        epoch_loss += loss.item()  #Ajout de la loss de ce batch

    avg_epoch_loss = epoch_loss / len(train_loader)
    scheduler.step(avg_epoch_loss) #Permet de diminuer le lr lorsque l'epoch stagne
    train_losses.append(avg_epoch_loss)  #Stocker la loss de l'époque (pour le graphique)

    print(f"Époque [{epoch+1}/75], Loss: {avg_epoch_loss:.4f}")

#Évaluation du modèle
best_model.eval()
with torch.no_grad():
    y_pred_test = best_model(X_test_tensor)
    y_pred_test = torch.sigmoid(y_pred_test).cpu().numpy()
    fpr, tpr, _ = roc_curve(y_test_tensor.cpu().numpy(), y_pred_test)
    roc_auc = auc(fpr, tpr)
    y_pred_binary = (y_pred_test >= 0.6).astype(int) #seuil à 0.6 au lieu de 0.5
    y_true = y_test_tensor.cpu().numpy().astype(int)
    

#Calcul des métriques
conf_matrix = confusion_matrix(y_true, y_pred_binary)
print("Matrice de Confusion:")
print(conf_matrix)

precision = precision_score(y_true, y_pred_binary)
recall = recall_score(y_true, y_pred_binary)
f1 = f1_score(y_true, y_pred_binary)
print(f"Précision : {precision:.4f}")
print(f"Rappel : {recall:.4f}")
print(f"Score F1 (1) : {f1:.4f}")
print("Classification Report:")
print(classification_report(y_true, y_pred_binary, digits=4))

#Courbe train loss permet d'optimiser le choix du nombre d'epoch
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Époques')
plt.ylabel('Loss')
plt.title('Courbe de Train Loss')
plt.legend()
plt.show()

#courbe ROC
print(f"AUC : {roc_auc:.4f}")
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taux de Faux Positifs (FPR)')
plt.ylabel('Taux de Vrais Positifs (TPR)')
plt.title('Courbe ROC')
plt.legend(loc='lower right')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

#Matrice de confusion
cm = confusion_matrix(y_true, y_pred_binary)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negatif", "Positif"], yticklabels=["Negatif", "Positif"])
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.title("Matrice de Confusion")
plt.show()
