# 🚩 Find The Flag — Landmass Prediction using Decision Trees & Random Forest

![Subject](https://img.shields.io/badge/Subject-Data%20Science%20%2F%20AI-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Decision%20Trees-purple)
![Type](https://img.shields.io/badge/Type-Supervised%20Classification-orange)
![Status](https://img.shields.io/badge/Status-Portfolio%20Ready-success)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-teal)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)

Projet de Machine Learning supervisé visant à prédire le continent d'appartenance (*Africa, Asia, Europe, North America, Oceania, South America*) d'un pays à partir des caractéristiques visuelles et géométriques de son drapeau.

---

## 📌 Vue d'ensemble

Le projet explore le pouvoir discriminant des symboles et couleurs héraldiques. Le jeu de données a été nettoyé des entités géopolitiques obsolètes (URSS, Yougoslavie, etc.), enrichi avec les nouveaux états indépendants (Soudan du Sud, Kosovo, Timor Oriental...) et mappé directement avec les noms explicites des continents.

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
   * Exploitation du dataset modernisé et actualisé (213 pays/territoires).
   * Extraction de 16 variables explicatives (7 couleurs + 9 motifs géométriques).
   * Utilisation directe de la cible textuelle `Landmass_Name` pour garantir une lisibilité directe des graphiques.
2. **Analyse du Biais / Variance :**
   * Évaluation des courbes d'exactitude Train vs Test sur une plage de profondeurs (`max_depth` de 1 à 20) pour identifier le seuil d'overfitting.
3. **Optimisation & Cross-Validation :**
   * Recherche par grille (`GridSearchCV`) sur 5 plissements stratifiés (`StratifiedKFold`).
   * Hyperparamètres optimaux retenus : `criterion='gini'`, `max_depth=7`, `min_samples_leaf=4`, `min_samples_split=10`.
4. **Modélisation d'Ensemble :**
   * Comparaison avec un algorithme `RandomForestClassifier` (100 arbres) évalué en 5-Fold Cross-Validation.
5. **Génération des Artefacts Visuels :**
   * Export automatique de `accuracy_vs_depth.png`, `confusion_matrix.png` et `decision_tree.png` intégrant les noms explicites de continents.

---

## 📁 Structure du Projet

```text
.
├── flags.csv                  # Dataset source modernisé & actualisé
├── script.py                  # Pipeline complet Machine Learning Python
├── index.html                 # Dashboard Web interactif de présentation
├── README.md                  # Documentation du projet
├── accuracy_vs_depth.png      # Graphique d'évaluation Train vs Test
├── confusion_matrix.png       # Matrice de confusion avec continents réels
└── decision_tree.png          # Visualisation haute résolution de l'arbre
```

---

## 📈 Métriques & Résultats

* **GridSearchCV (Decision Tree) :** 43.5% d'accuracy en 5-Fold Cross-Validation (`max_depth=7`, `min_samples_leaf=4`, `min_samples_split=10`).
* **Random Forest :** 39.9% (+/- 5.2%) d'accuracy moyenne en 5-Fold Cross-Validation.
* **Performance par Continent :** L'Europe est le continent le plus reconnaissable (F1-score de 0.57, Recall de 67%).

---

## 🏆 Importance des Features (Gini Importance)

1. **Stripes (16.7%)** — Structure de bandes horizontales (Amériques / Afrique).
2. **Sunstars (15.9%)** — Soleils & Étoiles (Asie, Amériques, Océanie).
3. **Green (12.2%)** — Ancrage culturel fort (Afrique, Monde Arabo-Musulman).
4. **White (8.9%)** — Couleur de structure/délimitation universelle.
5. **Gold / Yellow (8.3%)** — Marqueur géographique (Afrique, Amérique du Sud).
6. **Saltires (7.6%)** — Croix de Saint-André (Europe, Caraïbes).
7. **Circles (6.5%)** — Concentration en Asie et Océanie.
8. **Crosses (5.7%)** — Croix héraldiques européennes.
9. **Blue (5.0%)** — Dominante en Europe, Océanie et Amérique du Sud.
10. **Red (4.6%)** — Couleur omniprésente, peu discriminante seule.
11. **Black (4.2%)** — Couleur associée aux drapeaux africains.
12. **Bars (4.2%)** — Bandes verticales héraldiques européennes.

---

## 🚀 Installation & Lancement

1. **Installer les dépendances :**
   ```bash
   pip3 install pandas matplotlib seaborn scikit-learn
   ```

2. **Exécuter le pipeline ML :**
   ```bash
   python3 script.py
   ```

3. **Consulter le tableau de bord :**
   Ouvrir `index.html` dans un navigateur web.