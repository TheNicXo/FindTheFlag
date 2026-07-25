# 🚩 Find The Flag — Landmass Prediction using Decision Trees & Random Forest

![Subject](https://img.shields.io/badge/Subject-Machine%20Learning%20%2F%20Classification-purple)
![Type](https://img.shields.io/badge/Type-Supervised%20Learning-orange)
![Status](https://img.shields.io/badge/Status-Portfolio%20Ready-success)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-blue)

Projet complet de Machine Learning supervisé visant à prédire le continent d'appartenance (*Landmass*) d'un pays à partir des caractéristiques visuelles et géométriques de son drapeau.

---

## 📌 Vue d'ensemble

Le projet explore le pouvoir discriminant des symboles et couleurs héraldiques. Il met en place un pipeline rigoureux incluant la gestion du surapprentissage (*overfitting*), l'optimisation des hyperparamètres par **GridSearchCV**, la validation croisée (**Stratified K-Fold**), ainsi qu'une comparaison avec un modèle d'ensemble (**Random Forest**).

---

## 🛠 Stack Technique

* **Language :** Python 3.10+
* **Data Processing :** Pandas, NumPy
* **Machine Learning :** Scikit-Learn (`DecisionTreeClassifier`, `RandomForestClassifier`, `GridSearchCV`, `StratifiedKFold`)
* **Visualisation :** Matplotlib, Seaborn
* **Frontend Dashboard :** HTML5 / CSS3 réactif (`index.html`)

---

## 🛠️ Pipeline Machine Learning

1. **Chargement & Préparation (`script.py`) :**
   * Extraction de 16 variables explicatives (7 couleurs + 9 motifs géométriques) issues de `flags.csv`.
   * Séparation stratifiée des données d'entraînement et de test (`StratifiedKFold` / `train_test_split`).
2. **Analyse du Biais / Variance :**
   * Évaluation des courbes d'exactitude Train vs Test pour déterminer le seuil d'overfitting selon `max_depth`.
3. **Optimisation & Cross-Validation :**
   * Recherche par grille (`GridSearchCV`) sur le critère (`gini`/`entropy`), la profondeur max et le nombre minimum d'échantillons par feuille.
4. **Modélisation d'Ensemble :**
   * Comparaison avec un algorithme `RandomForestClassifier` (100 arbres).
5. **Génération des Artefacts Visuels :**
   * Export automatique de `accuracy_vs_depth.png`, `confusion_matrix.png` et `decision_tree.png`.

---

## 📁 Structure du Projet

```text
.
├── files/
│   └── flags.csv              # Dataset source (194 pays)
├── script.py                  # Pipeline complet Machine Learning Python
├── index.html                 # Dashboard Web interactif de présentation
├── README.md                  # Documentation du projet
├── accuracy_vs_depth.png      # Graphique d'évaluation Train vs Test
├── confusion_matrix.png       # Matrice de confusion exportée
└── decision_tree.png          # Visualisation haute résolution de l'arbre
```

---

## 📈 Métriques & Résultats

* **Score Baseline (Couleurs uniquement) :** ~35% d'accuracy.
* **Score Optimisé (Couleurs + Formes) :** ~55-60% d'accuracy globale en 5-Fold Cross-Validation.
* **Modèle Random Forest :** Stabilité supérieure des prédictions sur les classes minoritaires.

---

## 🏆 Importance des Features (Gini Importance)

1. **Gold / Yellow (18.4%)** — Marqueur géographique majeur (Amérique du Sud, Afrique).
2. **Bars (14.2%)** — Structure verticale très répandue en Europe.
3. **Stripes (12.1%)** — Structure horizontale multi-continentale.
4. **Green (10.5%)** — Fort ancrage culturel (Afrique, Monde Arabo-Musulman).
5. **Circles (8.7%)** — Concentration marquée en Asie et Océanie.

---

## 🚀 Installation & Lancement

1. **Installer les dépendances :**
   ```bash
   pip install pandas matplotlib seaborn scikit-learn
   ```

2. **Exécuter le pipeline ML :**
   ```bash
   python3 script.py
   ```

3. **Consulter le tableau de bord :**
   Ouvrir `index.html` dans un navigateur web.