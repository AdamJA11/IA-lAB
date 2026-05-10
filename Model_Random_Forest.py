import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Charger le dataset
df = pd.read_csv("train.csv")

print("Aperçu :")
print(df.head())

print("\nInfos :")
print(df.info())

# 2. Gérer les valeurs manquantes (Correction du FutureWarning)
# numériques → moyenne
for col in df.select_dtypes(include=np.number):
    df[col] = df[col].fillna(df[col].mean())

# catégorielles → valeur la plus fréquente
for col in df.select_dtypes(include="object"):
    df[col] = df[col].fillna(df[col].mode()[0])

# 3. Créer la target (Correction du KeyError : on utilise score_examen)
df["need_help"] = (df["score_examen"] < 50).astype(int)
# ==========================================
# 📊 EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
sns.set_theme(style="whitegrid")



# Graphique 1 : Distribution de la variable cible
plt.figure(figsize=(6, 4))
sns.countplot(x="need_help", data=df, palette="Set2")
plt.title("Repartition des elves (0 = OK, 1 = Besoin aide)")
plt.xlabel("Besoin aide (Target)")
plt.ylabel("Nombre étudiants")
plt.show()

# Graphique 2 : Matrice de corrélation (pour voir les liens entre les variables numériques)
plt.figure(figsize=(10, 8))
# On sélectionne uniquement les colonnes numériques pour la corrélation
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matrice de Correlation des feature ")
plt.show()

# Graphique 3 : Impact de l'assiduité sur l'échec (Boxplot)
if "assiduité_classe" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="need_help", y="assiduité_classe", data=df, palette="pastel")
    plt.title("Impact de l'assiduite sur le besoin aide")
    plt.xlabel("Besoin d'aide (0 = Non, 1 = Oui)")
    plt.ylabel("Assiduité en classe (%)")
    plt.show()



# 4. Supprimer la colonne cible originale ET l'ID
df = df.drop(columns=["score_examen", "id"], errors="ignore")

# 5. Encoder les variables catégorielles
df = pd.get_dummies(df, drop_first=True)

# 6. Séparer X et y
X = df.drop("need_help", axis=1)
y = df["need_help"]

# 7. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 8. ENTRAÎNEMENT ET SAUVEGARDE DES PRÉDICTIONS
# ==========================================
print("\n=== ENTRAÎNEMENT DES MODÈLES EN COURS ===")

# Dictionnaires pour stocker les résultats et les prédictions
accuracies = {}
predictions = {}

# 8.1 Modèle Naïf Intelligent
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

# 8.2 Modèles Machine Learning
models = {
    "Régression Logistique": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    "Deep Learning (MLP)": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
}

for name, model_pipeline in models.items():
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    accuracies[name] = accuracy_score(y_test, y_pred)
    predictions[name] = y_pred

print("Entraînement terminé !")

# ==========================================
# 📈 9. VISUALISATIONS COMPARATIVES (POUR LE RAPPORT)
# ==========================================
sns.set_theme(style="whitegrid")

# Graphique A : Comparaison des Accuracies (Barplot)
plt.figure(figsize=(10, 5))
ax = sns.barplot(x=list(accuracies.values()), y=list(accuracies.keys()), palette="viridis")
plt.title("Comparaison des Performances (Accuracy) des Modèles", fontsize=14)
plt.xlabel("Accuracy (Score global)")
plt.ylabel("Modèles")
plt.xlim(0, 1.05)

# Ajouter les chiffres sur les barres
for i, v in enumerate(accuracies.values()):
    ax.text(v + 0.01, i, f"{v:.4f}", color='black', va='center', fontweight='bold')
plt.show()

# Graphique B : Comparaison des Matrices de Confusion (Grille 2x2)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Comparaison des Matrices de Confusion", fontsize=16, fontweight='bold')

axes = axes.flatten() # Pour itérer facilement sur la grille 2x2

for i, (name, preds) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Prédit: OK', 'Prédit: Aide'],
                yticklabels=['Réel: OK', 'Réel: Aide'])
    axes[i].set_title(f"{name}\nAccuracy: {accuracies[name]:.4f}")
    axes[i].set_ylabel('Valeurs Réelles')
    axes[i].set_xlabel('Valeurs Prédites')

plt.tight_layout()
plt.subplots_adjust(top=0.90) # Ajuster l'espacement pour le titre principal
plt.show()

# ==========================================
# 🧠 10. IMPORTANCE DES VARIABLES (via Random Forest)
# ==========================================
rf_explainer = models["Random Forest"]
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_explainer.feature_importances_
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", data=feature_importance.head(10), palette="magma")
plt.title("Top 10 des variables les plus importantes (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Variables")
plt.show()
# ==========================================
# 🔥 11. Modèle naïf intelligent
# ==========================================
def naive_smart(X_data):
    pred = np.zeros(len(X_data))
    if "assiduité_classe" in X_data.columns:
        pred[X_data["assiduité_classe"] < 75] = 1
    if "heures_etude" in X_data.columns:
        pred[X_data["heures_etude"] < 2] = 1
    return pred

y_naive = naive_smart(X_test)
print("\n=== MODEL NAIF (INTELLIGENT) ===")
print("Accuracy:", accuracy_score(y_test, y_naive))


# ==========================================
# 📊 12. Comparaison Naïf vs Machine Learning
# ==========================================
print("\n=== COMPARAISON FINALE ===")
print("Naive accuracy:", accuracy_score(y_test, y_naive))
print(f"ML accuracy ({best_name}):", accuracy_score(y_test, y_pred))


# ==========================================
# 🧠 13. Importance des variables (via Random Forest)
# ==========================================
# Même si le Random Forest ne gagne pas, on l'entraîne rapidement pour extraire 
# l'importance des variables et pouvoir l'afficher dans le graphique.
rf_explainer = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_explainer.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_explainer.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n=== FEATURE IMPORTANCE (Extraite du Random Forest) ===")
print(feature_importance.head(10))