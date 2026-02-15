# Documentation technique du modèle

## Objectif

Prédire la probabilité qu'un client quitte l'entreprise (churn) à partir
de 61 variables numériques.

## Modèle

-   Algorithme : **Régression Logistique** (scikit-learn)
-   Sortie : `predict_proba(X)[:, 1]` = probabilité de churn
-   Seuil optimisé : **0.646** (maximisation du F1-score sur la classe
    churn)

## Interprétation

-   `proba >= 0.646` → label = 1 (client à risque)
-   `proba < 0.646` → label = 0 (client stable)

## Performances sur jeu de test

  Classe   Précision   Rappel   F1-score   Support
  -------- ----------- -------- ---------- ---------
  0        0.918       0.903    0.910      247
  1        0.529       0.574    0.551      47

-   Accuracy : 0.85
-   F1 macro : 0.731
-   F1 pondéré : 0.853

## Analyse

Le modèle présente d'excellentes performances sur la classe majoritaire.
La détection de la classe churn reste satisfaisante malgré le
déséquilibre des classes. Le seuil optimisé permet un compromis
pertinent entre précision et rappel.

## Maintenance

-   Réentraînement recommandé tous les 6 à 12 mois.
-   Surveillance du F1-score et du recall sur la classe churn.
-   Vérification régulière de la dérive des données.
