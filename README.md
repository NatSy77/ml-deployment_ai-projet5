---
title: "Déploiement ML – Futurisys"
emoji: "🚀"
colorFrom: "blue"
colorTo: "purple"
sdk: "docker"
app_port: 7860
pinned: false
---

# Déploiement d'un modèle de Machine Learning – Projet Futurisys  
Projet OpenClassrooms – *Déployez votre modèle de machine learning*

---

## Description du projet

Ce projet consiste à déployer en production le modèle de Machine Learning développé lors du **Projet 4 – Classification automatique d'informations**.  
Le modèle repose sur une **régression logistique** permettant de prédire la probabilité qu’un client quitte ou reste dans l’entreprise.
Le modèle de régression logistique atteint une accuracy de 85 % sur le jeu de test, avec un F1-score de 0.55 sur la classe churn, grâce à un seuil de décision optimisé.


L’objectif est de rendre ce modèle accessible via une **API FastAPI**, tout en respectant les bonnes pratiques d’ingénierie logicielle :

- API performante, documentée et conteneurisée avec Docker  
- Dépôt Git structuré et versionné  
- Tests unitaires, fonctionnels et d’intégration  
- Base de données PostgreSQL avec traçabilité complète  
- Pipeline CI/CD avec GitHub Actions  

Le projet est développé pour **Futurisys**, une entreprise fictive souhaitant industrialiser ses modèles ML.

---

## Architecture de la solution

- **FastAPI** : exposition du modèle via une API REST
- **Scikit-learn** : modèle de régression logistique
- **PostgreSQL** : stockage des données clients et de la traçabilité
- **SQLAlchemy** : ORM pour l’accès à la base
- **Docker** : conteneurisation de l’API
- **GitHub Actions** : intégration continue

---

## Installation et lancement

### 1. Cloner le projet

```bash
git clone https://github.com/NatSy77/ml-deployment_ai-projet5.git
cd ml-deployment_ai-projet5
```
### 2. Installer les dépendances

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### 3. Lancer l’API

uvicorn app.main:app --reload --port 7860
L’API sera disponible à l’adresse : http://localhost:7860

---

## Structure du dépôt

ml-deployment_ai-projet5/

│
├── app/
│ ├── main.py # Application FastAPI (endpoints)
│ ├── models_ml.py # Logique de prédiction ML
│ ├── models_db.py # Modèles SQLAlchemy
│ ├── schemas.py # Schémas Pydantic
│ ├── db.py # Connexion PostgreSQL
│ ├── crud.py # Accès base de données
│ └── config.py # Variables d'environnement
│
├── model/
│ └── model_project4.pkl # Modèle issu du projet 4
│
├── tests/
│ ├── test_api.py
│ ├── test_model.py
│ ├── test_schemas.py
│ ├── test_db.py
│ └── test_health.py
│
├── db/
│ ├── create_db.sql
│ └── samples/
│
├── requirements.txt
├── docker-compose.yml
├── README.md
└── pyproject.toml

---

## Intégration d’une base PostgreSQL et traçabilité

Une base de données PostgreSQL est utilisée afin de stocker :

- le dataset complet des clients (`clients`)
- les requêtes envoyées au modèle (`prediction_requests`)
- les prédictions générées (`prediction_outputs`)

### Principe de fonctionnement

L’endpoint `/predict` reçoit un `client_id`.

1. Les features sont récupérées depuis la table `clients`
2. L’input est enregistré dans `prediction_requests`
3. La prédiction est calculée par le modèle
4. L’output est enregistré dans `prediction_outputs`

Cela garantit une **traçabilité complète et auditabile** des prédictions du modèle.

### Exemple de vérification SQL

```sql
SELECT * FROM prediction_requests ORDER BY created_at DESC LIMIT 3;
SELECT * FROM prediction_outputs ORDER BY created_at DESC LIMIT 3;
```

---

## Tests et qualité du code

Une suite complète de tests a été mise en place afin de garantir la fiabilité, la robustesse et la reproductibilité de l’API et du modèle de Machine Learning.

### Types de tests

- **Tests unitaires**
  - Logique de prédiction du modèle (`predict_features`)
  - Validation des schémas Pydantic (types, longueurs, champs requis)

- **Tests fonctionnels**
  - Endpoint `/health`
  - Endpoint `/predict_raw` (cas nominal et erreurs)

- **Tests d’intégration (DB-first)**
  - Endpoint `/predict` avec lecture en base PostgreSQL
  - Vérification de la traçabilité (`prediction_requests`, `prediction_outputs`)
  - Ces tests sont exécutés uniquement lorsque la base PostgreSQL est disponible

### Lancer les tests

#### Tests unitaires et fonctionnels (sans base de données)
```bash
python -m pytest -q
```
#### Tests avec base PostgreSQL (Docker requis)
docker-compose up -d
python -m pytest tests/test_db.py -q

---

## Documentation

La documentation complète du projet est disponible dans le dossier `docs/` :

- `docs/api.md` : documentation détaillée des endpoints de l’API
- `docs/model.md` : documentation technique du modèle de Machine Learning

Une documentation interactive est également accessible via Swagger à l’adresse :

http://localhost:7860/docs
