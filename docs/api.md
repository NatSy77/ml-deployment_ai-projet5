# Documentation API (FastAPI)

## Accès à la documentation interactive

-   Swagger UI : `/docs`
-   OpenAPI : `/openapi.json`

------------------------------------------------------------------------

## Endpoints

### GET `/health`

Vérifie que l'API est opérationnelle.

**Réponse (200)**

``` json
{
  "status": "ok"
}
```

------------------------------------------------------------------------

### POST `/predict_raw`

Prédiction à partir d'un vecteur de features (utilisé pour les tests et
la CI, sans dépendance à la base de données).

**Payload**

``` json
{
  "features": [0.0, 0.0, "...", 0.0]
}
```

**Réponse (200)**

``` json
{
  "label": 0,
  "proba": 0.23,
  "threshold": 0.646
}
```

**Erreurs possibles** - `422` : payload invalide - `500` : erreur
interne

------------------------------------------------------------------------

### POST `/predict`

Prédiction DB-first : prend un `client_id`, récupère les features en
base PostgreSQL et enregistre automatiquement la traçabilité.

**Payload**

``` json
{
  "client_id": 123
}
```

**Réponse (200)**

``` json
{
  "label": 1,
  "proba": 0.84,
  "threshold": 0.646
}
```

**Erreurs possibles** - `404` : client_id introuvable - `422` : payload
invalide - `500` : problème base ou modèle

------------------------------------------------------------------------

## Exemple d'utilisation (curl)

``` bash
curl -X POST http://localhost:7860/predict_raw   -H "Content-Type: application/json"   -d '{"features":[0.0,0.0,...,0.0]}'
```
