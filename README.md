# Déploiement d'un modèle de Machine Learning – Projet Futurisys  
Projet OpenClassrooms – *Déployez votre modèle de machine learning*

---

## Description du projet

Ce projet consiste à déployer en production le modèle de Machine Learning développé lors du **Projet 4 – Classification automatique d'informations**.  
L’objectif est de rendre ce modèle accessible via une **API FastAPI**, tout en respectant les bonnes pratiques d’ingénierie logicielle :

- API performante, documentée et sécurisée  
- Dépôt Git structuré et versionné  
- Tests unitaires (Pytest)  
- Base de données PostgreSQL  
- Pipeline CI/CD avec GitHub Actions  

Le projet est développé pour **Futurisys**, une entreprise fictive souhaitant industrialiser ses modèles ML.

---

## Structure du dépôt

ml-deployment_ai-projet5/

│
├── app/
│ ├── main.py # Application FastAPI (+ endpoints)
│ ├── models_ml.py # Chargement du modèle ML
│ ├── schemas.py # Schémas Pydantic (requêtes & réponses)
│ ├── db.py # Connexion PostgreSQL
│ ├── crud.py # Fonctions d'accès BDD
│ └── config.py # Gestion des variables d'environnement
│

├── model/
│ └── model_project4.pkl # Modèle issu du projet 4
│

├── tests/
│ ├── test_api.py
│ ├── test_model.py
│ └── test_db.py
│

├── db/
│ ├── create_db.sql # Script de création de la base
│ └── samples/ # Exemples d'inputs/outputs du modèle
│

├── requirements.txt
├── README.md
├── .gitignore
└── pyproject.toml


---

## Installation

### 1. Cloner le projet
```bash
git clone https://github.com/NatSy77/ml-deployment_ai-projet5.git
cd ml-deployment_ai-projet5

