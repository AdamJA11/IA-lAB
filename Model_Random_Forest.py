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


df = pd.read_csv("train.csv")

print("Aperçu :")
print(df.head())

print("\nInfos :")
print(df.info())


for col in df.select_dtypes(include=np.number):
    df[col] = df[col].fillna(df[col].mean())


for col in df.select_dtypes(include="object"):
    df[col] = df[col].fillna(df[col].mode()[0])


df["need_help"] = (df["score_examen"] < 50).astype(int)

sns.set_theme(style="whitegrid")




plt.figure(figsize=(6, 4))
sns.countplot(x="need_help", data=df, palette="Set2")
plt.title("Repartition des elves (0 = OK, 1 = Besoin aide)")
plt.xlabel("Besoin aide (Target)")
plt.ylabel("Nombre étudiants")
plt.show()


plt.figure(figsize=(10, 8))

numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matrice de Correlation des feature ")
plt.show()


if "assiduité_classe" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="need_help", y="assiduité_classe", data=df, palette="pastel")
    plt.title("Impact de l'assiduite sur le besoin aide")
    plt.xlabel("Besoin d'aide (0 = Non, 1 = Oui)")
    plt.ylabel("Assiduité en classe (%)")
    plt.show()




df = df.drop(columns=["score_examen", "id"], errors="ignore")


df = pd.get_dummies(df, drop_first=True)


X = df.drop("need_help", axis=1)
y = df["need_help"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print("\n=== COMPARAISON DES MODÈLES (Cross-Validation 5-folds) ===")


models = {
    "Régression Logistique": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    "Deep Learning (MLP)": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
}

best_name = ""
best_score = 0
best_model = None


for name, model_pipeline in models.items():
    
    cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    mean_cv = cv_scores.mean()
    
    print(f"-> {name} | Accuracy moyenne (CV) : {mean_cv:.4f} (+/- {cv_scores.std() * 2:.4f})")
    
   
    if mean_cv > best_score:
        best_score = mean_cv
        best_name = name
        best_model = model_pipeline

print(f"\n🏆 Le modèle sélectionné est : {best_name} (Score CV : {best_score:.4f})")



best_model.fit(X_train, y_train)


y_pred = best_model.predict(X_test)

print("\n=== ÉVALUATION DU MEILLEUR MODÈLE ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))



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



print("\n=== COMPARAISON FINALE ===")
print("Naive accuracy:", accuracy_score(y_test, y_naive))
print(f"ML accuracy ({best_name}):", accuracy_score(y_test, y_pred))



rf_explainer = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_explainer.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_explainer.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n=== FEATURE de ouf  (Extraite du Random Forest) ===")
print(feature_importance.head(10))
print( "\n=== FEATURE de ouf  (Extraite du Random Forest) ===")