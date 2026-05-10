import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn - Preprocessing & Selection
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Sklearn - Modèles
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Sklearn - Métriques
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# ==========================================
# 1. CHARGEMENT ET NETTOYAGE
# ==========================================
df = pd.read_csv("train.csv")

# Gestion des valeurs manquantes (Imputation)
for col in df.select_dtypes(include=np.number):
    df[col] = df[col].fillna(df[col].mean())

for col in df.select_dtypes(include="object"):
    df[col] = df[col].fillna(df[col].mode()[0])

# Création de la cible (Target) : 1 si échec (< 50), 0 sinon
df["need_help"] = (df["score_examen"] < 50).astype(int)

# ==========================================
# 📊 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
sns.set_theme(style="whitegrid")

# Graphique 1 : Distribution de la cible (Déséquilibre des classes)
plt.figure(figsize=(6, 4))
sns.countplot(x="need_help", data=df, palette="Set2")
plt.title("Répartition : 0 = Réussite, 1 = Besoin d'aide")
plt.show()

# Graphique 2 : Matrice de corrélation
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Matrice de Corrélation des Variables")
plt.show()

# ==========================================
# 3. PRÉPARATION POUR LE MACHINE LEARNING
# ==========================================
# Supprimer les colonnes inutiles et la fuite de données (score_examen)
df = df.drop(columns=["score_examen", "id"], errors="ignore")

# Encodage des variables catégorielles (One-Hot)
df = pd.get_dummies(df, drop_first=True)

# Séparation X (features) et y (target)
X = df.drop("need_help", axis=1)
y = df["need_help"]

# Découpage Train (80%) / Test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 4. COMPARAISON DES 3 MODÈLES + NAÏF
# ==========================================
print("\n=== PHASE D'ENTRAÎNEMENT ET COMPARAISON ===")

# Dictionnaires pour stocker les résultats
accuracies = {}
predictions = {}
models_list = {
    "Régression Logistique": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    "Deep Learning (MLP)": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
}

# 4.1 Modèle Naïf Intelligent (Baseline)
def naive_smart(X_data):
    pred = np.zeros(len(X_data))
    if "assiduité_classe" in X_data.columns:
        pred[X_data["assiduité_classe"] < 75] = 1
    if "heures_etude" in X_data.columns:
        pred[X_data["heures_etude"] < 2] = 1
    return pred

y_naive = naive_smart(X_test)
accuracies["Modèle Naïf"] = accuracy_score(y_test, y_naive)
predictions["Modèle Naïf"] = y_naive

# 4.2 Entraînement des modèles ML avec Cross-Validation
best_model_name = ""
best_score = 0

for name, model in models_list.items():
    # Cross-Validation pour la robustesse (sur X_train pour ne pas polluer le test)
    cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()
    print(f"-> {name} | Accuracy CV : {cv_score:.4f}")
    
    # Entraînement final sur le train set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracies[name] = accuracy_score(y_test, y_pred)
    predictions[name] = y_pred
    
    if cv_score > best_score:
        best_score = cv_score
        best_model_name = name

print(f"\n🏆 Meilleur modèle sélectionné : {best_model_name}")

# ==========================================
# 📈 5. VISUALISATIONS DES RÉSULTATS (POUR LE RAPPORT)
# ==========================================

# Graphique A : Barplot de comparaison des Accuracies
plt.figure(figsize=(10, 5))
colors = ['grey', 'skyblue', 'salmon', 'lightgreen']
ax = sns.barplot(x=list(accuracies.values()), y=list(accuracies.keys()), palette=colors)
plt.title("Comparaison de l'Accuracy Globale", fontsize=14)
plt.xlabel("Score Accuracy")
plt.xlim(0, 1.0)
for i, v in enumerate(accuracies.values()):
    ax.text(v + 0.01, i, f"{v:.4f}", va='center', fontweight='bold')
plt.show()

# Graphique B : Grille des Matrices de Confusion
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Comparaison des Matrices de Confusion", fontsize=16, fontweight='bold')
axes = axes.flatten()

for i, (name, preds) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Prédit: OK', 'Prédit: Aide'],
                yticklabels=['Réel: OK', 'Réel: Aide'])
    axes[i].set_title(f"{name}\nAccuracy: {accuracies[name]:.4f}")

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()

# Graphique C : Importance des variables (via Random Forest)
rf_model = models_list["Random Forest"]
importances = rf_model.feature_importances_
feat_imp = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", data=feat_imp.head(10), palette="viridis")
plt.title("Top 10 : Pourquoi les élèves échouent ?")
plt.show()

# Graphique D : Courbe ROC du meilleur modèle
plt.figure(figsize=(8, 6))
best_pipeline = models_list[best_model_name]
y_prob = best_pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f'ROC {best_model_name} (AUC = {auc(fpr, tpr):.2f})', color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title("Courbe ROC - Capacité de Distinction")
plt.xlabel("Taux de Faux Positifs")
plt.ylabel("Taux de Vrais Positifs")
plt.legend()
plt.show()