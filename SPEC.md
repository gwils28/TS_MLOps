# SPEC W47 — protocole pré-enregistré

> **Statut : figé avant la première prévision** (commit `docs: pre-register W47 protocol`).
> Tout écart ultérieur est consigné dans [`IMPLEMENTATION_NOTES.md`](IMPLEMENTATION_NOTES.md)
> et signalé dans le rapport là où il intervient. Les valeurs numériques vivent dans
> [`configs/protocol.yaml`](configs/protocol.yaml) ; ce document en donne les raisons.

## 0. La question

Un forecaster de demande horaire est en production. La vérité terrain d'une prévision arrive
**après son horizon**. Peut-on mesurer la santé du modèle **sans attendre les labels**, et
**où cette mesure devient-elle fausse** ?

Hypothèse à tester (celle du brief) : l'estimation de performance sans labels (NannyML DLE)
suit l'erreur réalisée sous *covariate shift* et la rate sous *concept drift* pur.

Hypothèses supplémentaires, posées **avant** les résultats (je les écris pour pouvoir avoir tort) :

- **H-support.** DLE ne suit le covariate shift que *dans le support* de la référence. Le nanny
  model est un arbre : hors support (niveaux de demande jamais vus, cas COVID), il extrapole à
  plat et sous-estime l'erreur. Le choc COVID n'est donc pas « un choc covariable » au sens où
  le brief l'entend : il mêle concept drift, covariate shift hors support et covariate shift
  dans le support, selon la phase.
- **H-AR.** Pour un modèle autorégressif dont tous les lags sont ≥ H, un choc *né dans la cible*
  n'entre dans les features qu'après H heures. Pendant ces H heures, c'est un concept drift pur
  du point de vue du modèle : DLE est aveugle sur les prévisions déjà émises (« en vol »).
- **H-horloge.** Le gain de DLE sur le monitoring réalisé, mesuré en **temps d'horloge**
  (wall-clock), vaut au plus H + L (L = latence des labels après la date cible). Pour un choc né
  dans la cible et L = 0, ce gain est nul : le réalisé voit le choc dès le premier jour cible
  touché. Le gain réel de DLE vient (a) des covariables exogènes connues à l'avance, (b) de la
  latence L des labels, (c) de la quantification des prévisions en vol.

## 1. Données

- **Source :** NYC TLC, *yellow taxi trip records*, fichiers parquet mensuels
  (`https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet`),
  juillet 2016 → décembre 2021 (66 fichiers).
- **Traçabilité :** chaque fichier téléchargé est haché (SHA-256) dans
  `configs/data_manifest.csv` (avec nombre d'enregistrements bruts et conservés). Un
  re-téléchargement qui ne reproduit pas le hachage fait échouer le pipeline. Les fichiers bruts
  (~6 Go) ne sont pas versionnés ; la série horaire agrégée (`data/processed/yellow_hourly.csv`,
  48 240 lignes) l'est.
- **Agrégation :** nombre de prises en charge (`tpep_pickup_datetime`) par heure locale
  new-yorkaise. Seuls les enregistrements dont l'horodatage tombe dans le mois du fichier sont
  gardés (les fichiers TLC contiennent quelques dates aberrantes).
- **Heure d'été :** les horodatages TLC sont naïfs (heure locale). L'heure 02:00 du passage à
  l'heure d'été n'existe pas (≈ 0 trajet) : remplacée par la moyenne de ses voisines. L'heure
  01:00 du retour à l'heure d'hiver compte deux heures réelles : divisée par deux. Chaque
  réparation est journalisée (`results/tables/dst_repairs.csv`).
- **Pourquoi les taxis jaunes seuls :** série longue, homogène, publique, avec un choc réel
  documenté. Biais connu : tendance baissière structurelle 2016-2019 (concurrence VTC), soit
  une dérive lente *réelle* que le monitoring verra aussi (et c'est voulu).

## 2. Le modèle de temps (le point le plus important)

Trois horodatages par prévision, à ne jamais confondre :

| Symbole | Sens |
|---|---|
| τ (émission) | instant où la prévision est émise : τ = t − H |
| t (cible) | heure prévue |
| t + L (disponibilité du label) | instant où la vérité y_t entre dans le système |

- **Horizon H = 168 h** (une semaine) : la décision opérationnelle (planification de flotte,
  staffing) se prend une semaine avant. Choisir H = 1 semaine plutôt que 24 h rend la « fenêtre
  aveugle » assez longue pour être mesurée.
- **Contrainte d'information :** toute feature d'une cible t doit être observable à τ = t − H,
  donc **tout lag ≥ 168 h**. Testé automatiquement (`tests/test_features.py` corrompt le futur
  et vérifie que les features ne bougent pas).
- **Latence L :** 0 par défaut (la vérité de t est connue à t, cas optimiste). Le banc de
  détection rapporte aussi L = 168 h (labels consolidés une semaine après).
- **Disponibilité des signaux par chunk journalier (jour cible d) :**
  - estimation DLE, dérives d'entrées et de prédictions : dès que la dernière prévision du jour d
    est émise, soit fin du jour d − 7 ;
  - performance réalisée : fin du jour d + L.

Chaque délai de détection est rapporté **sur deux axes** : en temps cible (quel jour cible est le
premier signalé) et en temps d'horloge (quand l'alerte est disponible).

## 3. Découpage temporel (intervalles semi-ouverts)

| Période | Dates | Rôle |
|---|---|---|
| train | 2016-08-01 → 2018-07-01 | entraîne le forecaster (juillet 2016 sert aux lags) |
| reference | 2018-07-01 → 2019-07-01 | prévisions **hors échantillon**, labels connus : ajuste DLE, seuils et dérives |
| control | 2019-07-01 → 2020-03-01 | période sans choc connu : taux de fausses alertes, base des injections |
| covid | 2020-03-01 → 2022-01-01 | choc réel et reprise |

Pourquoi une référence hors échantillon : NannyML apprend l'erreur sur la référence. Si la
référence a servi à entraîner le forecaster, les erreurs y sont optimistes et toute estimation
ultérieure est biaisée vers le bas. Pourquoi une année complète : sinon les features
calendaires (mois) sortent du support de la référence dans l'analyse et chaque détecteur
univarié alerte sur la saison.

Phases du choc COVID (fixées par le calendrier, pas par les résultats) :

| Phase | Dates | Repère |
|---|---|---|
| P0 pré-choc | 2019-07-01 → 2020-03-09 | contrôle |
| P1 transition | 2020-03-09 → 2020-03-30 | chute (état d'urgence NY le 7 mars, *NY PAUSE* le 22 mars) |
| P2 confinement | 2020-03-30 → 2020-06-08 | creux (phase 1 de réouverture de NYC le 8 juin 2020) |
| P3 reprise | 2020-06-08 → 2021-07-01 | remontée progressive |
| P4 nouveau régime | 2021-07-01 → 2022-01-01 | niveau et profil hebdomadaire durablement différents |

## 4. Le forecaster

- LightGBM, cible `log1p(trips)`, prévision rétro-transformée en trajets ; hyperparamètres fixés
  a priori dans `protocol.yaml` (pas de tuning : le sujet est le monitoring, pas la précision).
- **Features** (toutes disponibles à τ = t − 168 h) :
  - lags de la même heure : `lag_168`, `lag_336`, `lag_504`, `lag_672` ; `lag_192` (8 jours) ;
  - `mean_same_hour_4w` = moyenne des quatre lags hebdomadaires ;
  - `roll24_at168`, `roll168_at168` : moyennes des 24 h et 168 h se terminant à t − 168 ;
  - `trend_ratio` = `roll168_at168` / moyenne des 168 h se terminant à t − 336 ;
  - calendrier (exogène, connu à l'avance) : `hour`, `dayofweek`, `month`, `is_holiday`
    (jours fériés fédéraux US via la librairie `holidays`).
- **Garde-fou de crédibilité :** sur la référence, le modèle doit battre la naïve saisonnière
  hebdomadaire (`lag_168`) en MAE. Sinon le monitoring d'un modèle inutile n'a pas de sens et je
  m'arrête pour corriger.

## 5. Le monitoring (NannyML 0.13.1)

Chunks **journaliers** (24 heures cibles), variante hebdomadaire en sensibilité.

| Signal | Méthode | Seuil (défaut NannyML sauf mention) |
|---|---|---|
| Performance estimée | DLE, MAE et RMSE | moyenne ± 3 σ de la performance réalisée des chunks de référence |
| Performance réalisée | `PerformanceCalculator` | idem |
| Dérive univariée | KS et Wasserstein (continues), χ² et Jensen-Shannon (catégorielles) | moyenne + 3 σ de la statistique sur les chunks de référence (χ² : p < 0,05) |
| Dérive multivariée | reconstruction PCA (65 % de variance) | moyenne ± 3 σ de l'erreur de reconstruction de référence |
| Dérive des prédictions | KS et Wasserstein sur `y_pred` | moyenne + 3 σ |
| Support | part des lignes dont une feature continue sort de [min, max] de la référence | 5 % (choix W47) |

**Réplique transparente de DLE.** Je réimplémente DLE en ~30 lignes (LightGBM sur |y − ŷ| et
(y − ŷ)², entrées = features + ŷ) et vérifie qu'elle reproduit NannyML. Puis une variante
**cross-fittée** (nanny model ajusté hors échantillon sur la référence) pour mesurer le biais
d'optimisme de l'ajustement en échantillon que fait NannyML (constaté en lisant son code :
`LGBMRegressor()` par défaut entraîné et évalué sur la même référence).

## 6. Expériences

### E1 — Choc réel (COVID)

Analyse continue 2019-07-01 → 2021-12-31. Sorties : courbes estimée vs réalisée (MAE, RMSE),
dérives, support ; tableau par phase : ratio estimé / réalisé, corrélation de rang, taux
d'accord des alertes. **Critère du brief :** l'estimée suit la réalisée là où le décalage est
covariable et dans le support ; je nomme chaque régime où elle décroche.

### E2 — Scénarios injectés (laboratoire)

Sur la période *control*, avec prévisions réelles du modèle figé. Chaque run tire un jour
d'apparition o ; fenêtre [o − 14 j, o + 28 j). Quatre familles :

| Famille | Construction | Ce qui change | Attendu pour DLE |
|---|---|---|---|
| **CS-harm** covariate shift nocif | après o, chaque jour = 24 lignes (x, y, ŷ) tirées dans le pool avec poids ∝ exp(β·z(x)), z = `lag_168` standardisé | P(x) seul ; P(y∣x) inchangé **par construction** (les couples sont réels, le poids ne dépend que de x) | suit |
| **CS-benign** covariate shift bénin | idem avec z = −`lag_168` (heures creuses) | P(x) seul, l'erreur absolue baisse | suit, et **ne doit pas alerter** |
| **CD-bias** concept drift pur | y′ = (1 + δ)·y après o ; x et ŷ gelés | P(y∣x) seul | aveugle |
| **CD-noise** concept drift pur (variance) | y′ = y·exp(s·ε − s²/2), ε ~ N(0,1) ; x et ŷ gelés | P(y∣x) seul (bruit aléatoire) | aveugle |
| **PROP** choc propagé (réaliste) | série × (1 + δ) dès o, features et prévisions **recalculées** | P(y∣x) d'abord, puis P(x) après 168 h | aveugle 7 jours, puis partiellement |

Le gel des features (CD-*) est une idéalisation assumée : il simule un facteur externe non
observé qui touche la cible sans passer par les entrées. PROP montre ce qui se passe quand ce
facteur se propage dans les lags, ce qui est le cas réel d'un modèle autorégressif.

Les runs *control* (aucune injection, mêmes fenêtres) donnent le taux de fausses alertes.

### E3 — Banc recall × délai (à la DriftGuard)

Pour chaque détecteur et chaque famille : **rappel** (part des runs où une alerte survient dans
[o, o + 28 j)), **délai médian** (temps cible et horloge, L ∈ {0, 168 h}), **taux de fausses
alertes** (alertes par 30 jours sur les runs control et avant o). Deux notions de « positif » :
*dérive* (toute injection) et *dommage* (l'erreur réalisée moyenne post-o dépasse le seuil
supérieur de référence) — un bon système d'alerte maximise le rappel du dommage, pas de la dérive.

Détecteurs comparés : dérive univariée seule (≥ 1 feature, ≥ 2 features), dérive multivariée
seule, dérive des prédictions, DLE seul, **règle jointe** (dérive d'entrée ET DLE au-dessus du
seuil, persistance 2 chunks), réalisé seul.

### E4 — Sensibilités et vérifications

Chunks hebdomadaires ; DLE cross-fitté ; réplique DLE vs NannyML ; seuils ± 2 σ / ± 3 σ.

### E5 — Réentraîner trop tôt (exploratoire, signalé comme tel)

Après le choc : réentraîner à o + k semaines avec les k semaines post-choc (k = 1, 2, 4, 8)
contre ne rien faire, évalué sur les 8 semaines suivantes. Illustre « réentraîner sur trop peu de
données peut empirer ». Exploratoire : pas de conclusion générale.

## 7. Ce qui est figé, ce qui ne l'est pas

Figé : données, découpage, features, hyperparamètres, seuils, familles d'injection et leurs
amplitudes, définitions de rappel / délai / fausse alerte, phases COVID.
Non figé (et signalé comme exploratoire s'il est utilisé) : toute variante ajoutée après avoir vu
les résultats.

## 8. Critères de fin

- `make test` vert (dont les tests anti-fuite).
- Le forecaster bat la naïve saisonnière sur la référence.
- La réplique DLE reproduit NannyML (écart relatif < 1 % sur la MAE estimée par chunk).
- Le rapport nomme chaque régime où DLE décroche, chiffres à l'appui, et le banc E3 donne
  recall × délai × fausses alertes pour chaque détecteur.
