# W47 · TS_MLOps — monitoring sans labels d'un forecaster de demande

Objectif : estimer la santé d'un forecaster horaire (NYC TLC, H = 168 h) sans labels avec
NannyML DLE, puis vérifier contre l'erreur réalisée et nommer les régimes où l'estimation ment.
Le protocole est figé dans `SPEC.md` ; les écarts vont dans `IMPLEMENTATION_NOTES.md`.

## Commandes
- `uv sync` puis `uv run pytest -q` (rapide, < 1 min) ; `make check` = ruff + tests.
- `make all` = données → prévisions → E1 → E2/E3 → figures → rapport.
- `make data` télécharge ~6 Go (66 fichiers) et vérifie les SHA-256 du manifeste ;
  la série agrégée `data/processed/yellow_hourly.csv` est versionnée, donc inutile en général.
- `make report` rend `report/` (Quarto ≥ 1.6) dans `docs/` (GitHub Pages).

## Pièges
- Trois horloges : émission τ = t − H, cible t, label t + L. Tout lag ≥ 168 h.
  Un délai de détection se rapporte toujours sur les deux axes (cible et horloge).
- La référence NannyML doit être hors échantillon pour le forecaster (sinon DLE est optimiste).
- NannyML 0.13 exige pandas < 3 et statsmodels ; kaleido est épinglé à 0.2.1 (roue x86_64).
- Horodatages TLC naïfs en heure locale : réparations DST dans `data.repair_dst`.
- Les chiffres du rapport viennent de `results/` (générés), jamais tapés à la main.

## Façon de travailler
- Rapport et docs en français ; code, identifiants et docstrings en anglais.
- Conventional Commits, un sujet par commit, sans ligne de co-auteur.
- Une variante ajoutée après avoir vu un résultat est étiquetée « exploratoire ».

## Compaction
Conserver : fichiers modifiés, commandes de test, décisions de modélisation, écarts au SPEC.
