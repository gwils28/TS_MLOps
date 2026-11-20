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

### D2 — MAPE ajoutée à DLE (exploratoire)

Le SPEC prévoyait MAE et RMSE. Pendant le confinement, la MAE réalisée *baisse* (168 trajets,
contre 812 avant le choc) parce que la demande s'effondre : une métrique absolue confond
« meilleur modèle » et « moins de volume ». J'ai ajouté la MAPE (supportée par DLE) comme
métrique sans échelle. Ajout fait **après** avoir vu les premières courbes E1 : exploratoire.

## Pièges et corrections

### P1 — NannyML décide catégoriel / continu par le dtype

`nannyml.base._split_features_by_type` : est catégorielle une colonne de dtype `object`,
`string`, `category` ou `bool`. Les colonnes calendaires `int` (`hour`, `dayofweek`, `month`,
`is_holiday`) étaient donc traitées comme **continues** par DLE (nanny model) et par la
reconstruction PCA, sans avertissement, alors que la dérive univariée les traitait comme
catégorielles (argument explicite `treat_as_categorical`). Symptôme : ma réplique DLE (qui passe
ces colonnes en catégoriel à LightGBM) s'écartait de NannyML de 15 % en médiane. Correction :
conversion en `category` avant tout appel NannyML (`monitoring._typed`). Après correction, la
réplique reproduit NannyML à 0,0 % près (critère SPEC §8).

### P2 — NannyML envoie de la télémétrie d'usage par défaut

Au premier appel, NannyML tente d'envoyer des événements d'usage à `api.segment.io` (bloqué ici
par le proxy, d'où des erreurs dans les journaux). En production ou chez un client, c'est une
fuite de métadonnées à couper : `NML_DISABLE_USAGE_LOGGING=1`, posé en tête de
`w47/monitoring.py`.

### P3 — Dépendances NannyML 0.13.1

- `pandas` 3 se résout mais n'est pas validé par NannyML : épinglé `< 3`.
- `statsmodels` est importé par `nannyml.plots` sans être déclaré : ajouté.
- `kaleido 0.2.1.post1` n'a de roue que pour armv7l : épinglé à `0.2.1`.

### P4 — Variante cross-fittée de DLE

Première version : la variante cross-fittée réutilisait le modèle plein sur l'analyse (seules
ses estimations de référence étaient hors échantillon), donc identique à NannyML hors
référence. Corrigé : sur l'analyse, elle moyenne les cinq modèles de pli.
