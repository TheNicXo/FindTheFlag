import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def load_and_preprocess_data(filepath: str):
    """Charge le dataset et prépare les variables explicatives et la cible."""
    df = pd.read_csv(filepath, header=0)
    
    y = df["Landmass"]
    
    color_features = ["Red", "Green", "Blue", "Gold", "White", "Black", "Orange"]
    shape_features = [
        "Bars", "Stripes", "Circles", "Crosses", "Saltires", 
        "Quarters", "Sunstars", "Crescent", "Triangle"
    ]
    extended_features = color_features + shape_features
    
    return df, df[extended_features], y, extended_features


def evaluate_decision_tree_depths(X, y):
    """Analyse l'impact de max_depth pour identifier l'overfitting (Train vs Test)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )
    
    train_scores, test_scores = [], []
    depths = range(1, 21)
    
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=1)
        clf.fit(X_train, y_train)
        train_scores.append(clf.score(X_train, y_train))
        test_scores.append(clf.score(X_test, y_test))
        
    plt.figure(figsize=(10, 5))
    plt.plot(depths, train_scores, label="Train Accuracy", marker="o")
    plt.plot(depths, test_scores, label="Test Accuracy", marker="s")
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    plt.title("Évolution de la précision selon max_depth (Train vs Test)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("accuracy_vs_depth.png", dpi=300)
    plt.close()


def optimize_hyperparameters_gridsearch(X, y):
    """Recherche des meilleurs hyperparamètres via GridSearchCV & Cross-Validation."""
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": range(1, 15),
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=1),
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )
    grid_search.fit(X, y)
    
    print(f"\n[GridSearchCV] Meilleurs paramètres Arbre : {grid_search.best_params_}")
    print(f"[GridSearchCV] Meilleure Accuracy (CV) : {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def compare_with_random_forest(X, y):
    """Entraîne et évalue un RandomForestClassifier via Cross-Validation."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=1)
    
    rf_scores = cross_val_score(rf_clf, X, y, cv=cv, scoring="accuracy")
    print(f"\n[Random Forest] Mean Accuracy (5-Fold CV) : {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")
    
    return rf_clf


def plot_tree_and_metrics(model, X_train, X_test, y_train, y_test, feature_names):
    """Affiche le rapport de classification, la matrice de confusion et l'arbre de décision."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    classes = [str(c) for c in sorted(y_train.unique())]
    
    # Rapport de classification
    print("\n--- Rapport de Classification ---")
    print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))
    
    # Affichage Importance des Features
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\n--- Importance des Features (Gini Importance) ---")
    for feat, val in importances.items():
        print(f"{feat:12s}: {val*100:.1f}%")

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Matrice de Confusion — Arbre de Décision")
    plt.xlabel("Classes Prédites")
    plt.ylabel("Classes Réelles")
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Visualisation de l'arbre
    plt.figure(figsize=(22, 12))
    plot_tree(
        model, 
        feature_names=feature_names, 
        class_names=classes, 
        filled=True, 
        rounded=True,
        fontsize=10
    )
    plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # 1. Chargement & Préparation
    df, X, y, feature_names = load_and_preprocess_data("flags.csv")
    
    # 2. Séparation Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )
    
    # 3. Évaluation Train vs Test
    evaluate_decision_tree_depths(X, y)
    
    # 4. Optimisation via GridSearchCV (Cross-Validation)
    best_dt_model = optimize_hyperparameters_gridsearch(X_train, y_train)
    
    # 5. Comparaison avec Random Forest
    compare_with_random_forest(X, y)
    
    # 6. Évaluation finale & Génération des graphiques
    plot_tree_and_metrics(best_dt_model, X_train, X_test, y_train, y_test, feature_names)