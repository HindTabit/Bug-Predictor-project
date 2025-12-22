# BugPredictor Pro 🐞

**Prédiction automatique des fichiers à risque dans les projets logiciels**

Projet de Génie Logiciel – Master S3 – 2025  
Prédiction de défauts logicielles à l'aide de métriques statiques et d'un modèle XGBoost entraîné sur des données réelles.

## Fonctionnalités principales

- **Copier-coller du code** : Analyse instantanée d'un extrait de code (Python, Java, JavaScript, C/C++)
- **Saisie manuelle des métriques** : Test rapide avec valeurs personnalisées
- **Upload d'un CSV** : Prédiction en batch sur des métriques extraites (ex: via lizard)
- **Analyse en direct d'un dépôt GitHub/GitLab** : Clone, extraction des métriques avec lizard, prédiction et classement des fichiers par risque

Modèle entraîné sur des métriques OO et procédurales (LOC, complexité cyclomatique, WMC, CBO, DIT, RFC, etc.).

## Démo en ligne

Une version en ligne est disponible ici :  
🔗 [[https://ton-app-streamlit.streamlit.app](https://ton-app-streamlit.streamlit.app](https://hindtabit-bug-predictor-project-appweb-sm02zq.streamlit.app))](https://hindtabit-bug-predictor-project-appweb-sm02zq.streamlit.app)  

## Installation locale

### Prérequis

- Python 3.8 ou supérieur
- Git

### Étapes

1. Cloner le dépôt

```bash
git clone [https://github.com/ton-username/Bug_Predictor-project.git](https://github.com/HindTabit/Bug-Predictor-project)
cd Bug_Predictor-project
Bash
2. Installer les dépendances

pip install -r requirements.txt

3. Ouvrire le dossier app : cd app

4. Lancer l'application

streamlit run app/web.py

L'application s'ouvre automatiquement dans votre navigateur à l'adresse :
http://localhost:8501
