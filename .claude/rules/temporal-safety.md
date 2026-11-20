# Sécurité temporelle (s'applique à tout le dépôt)

- Une feature de la cible t n'utilise que des valeurs ≤ t − 168 h. Un nouveau feature doit
  passer `tests/test_features.py` (corruption du futur).
- Intervalles de dates semi-ouverts [début, fin), horodatages naïfs heure locale NY.
- Le forecaster ne voit jamais la référence ni l'analyse à l'entraînement ; NannyML ne voit jamais
  l'analyse à l'ajustement.
- Les injections (E2) ne pondèrent que par x ; jamais par y ni par l'erreur.
