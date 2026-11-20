# Notes d'implémentation W47

Écarts au protocole pré-enregistré ([`SPEC.md`](SPEC.md)), pièges rencontrés et choix
conservateurs, dans l'ordre où ils sont apparus. Chaque écart est étiqueté (D1, D2…) et repris
dans le rapport là où il intervient.

## Écarts au SPEC

### D1 — Cible relative au niveau (garde-fou du forecaster déclenché)

- **Constat.** Le forecaster v0 pré-enregistré (cible `log1p(trips)`, lags bruts) **perd** contre
  la moyenne « même heure sur 4 semaines » sur la référence : MAE 1 015 contre 998, avec un biais
  de +506 trajets (sur-prévision). Le garde-fou de la SPEC §4 impose de s'arrêter et de corriger.
- **Cause.** La série baisse d'environ 15 % par an (concurrence VTC). Un arbre apprend des
  niveaux absolus : en 2018-2019, les lags tombent dans des feuilles apprises sur 2016-2017, plus
  hautes. Le feature `month`, vu seulement deux fois à l'entraînement, encode en partie l'année.
- **Correction.** Cible `log1p(y_t) − log1p(niveau)`, niveau = moyenne des 168 h se terminant à
  t − 168 (observable à l'émission) ; lags exprimés relativement à ce niveau (`rel_*`) ;
  `trend_ratio` et calendrier inchangés. Le niveau absolu n'entre plus dans le modèle mais reste
  **surveillé** (`roll168_at168` dans la dérive univariée).
- **Résultat sur la référence** (`results/tables/guard_reference.csv`) : MAE 825, biais −59,
  soit −17 % contre la moyenne 4 semaines. Trois variantes ont été essayées, sur la référence
  uniquement : avec niveau absolu en feature (MAE 925), sans niveau (825, retenue), sans `month`
  (1 092).
- **Ce qu'il faut savoir en lisant les résultats.** La décision a été prise sur la référence. La
  même impression écran affichait aussi les métriques v0 sur *control* et *covid* (v0 : MAE 1 464
  sur control, contre 1 092 pour la naïve) : je les ai vues, elles n'ont pas guidé le choix, mais
  je le signale.
- **Conséquence pour le monitoring.** Un modèle normalisé « cache » les chocs de niveau à ses
  propres entrées : seules la prédiction (absolue) et la variable de niveau surveillée les voient.
  C'est un résultat en soi, discuté dans le rapport (E1).
