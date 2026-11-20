# Bloc 1 — Veille : NannyML DLE et DriftGuard, lus à la source

> Sources lues le 20 novembre 2026. Notes détaillées avec citations verbatim, URL et numéros de
> page : [`source_notes.md`](source_notes.md). Le code de NannyML 0.13.1 a été lu directement
> dans le paquet installé (chemins cités ci-dessous).

## En bref

| Affirmation du brief | Verdict | Ce que disent les sources |
|---|---|---|
| NannyML estime la performance via CBPE (classification) ou DLE (régression), « n'importe quelle métrique, ex. RMSE » | **Imprécis** | « N'importe quelle métrique » vaut pour CBPE et les métriques de matrice de confusion. DLE couvre **six** métriques fixes : MAE, MAPE, MSE, RMSE, MSLE, RMSLE. |
| Dérive multivariée par reconstruction PCA et suivi de l'erreur de reconstruction | **Confirmé** | Imputation, encodage fréquentiel, standardisation, PCA à 65 % de variance, distance euclidienne moyenne par chunk. |
| DLE entraîne un nanny model qui estime la perte observation par observation | **Confirmé (verbatim)** | « train a nanny model that directly estimates the value of the loss of the monitored model for each observation ». |
| DriftGuard : > 30 000 séries M5, 97,8 % de rappel en 4,2 jours | **Verbatim, mais peu fiable** | Les chiffres sont dans l'abstract ; le papier en donne d'autres ailleurs (98 % / 5 j médian, 3 j, 5,1 j) et sa dérive est injectée dans la fenêtre d'entraînement M5. |
| Les liens entre crochets du brief | **Ne soutiennent pas les affirmations** | Deux pointent vers des papiers de contrôle synthétique, trois vers des billets sur Claude (hôtes bloqués, jugés sur leur URL). |

---

## 1. NannyML : ce que fait vraiment DLE

### 1.1 Le mécanisme (documentation + code)

La documentation (*How it works › Performance estimation*) décrit trois étapes ; le code
(`nannyml/performance_estimation/direct_loss_estimation/metrics.py`) les confirme et précise ce
que la documentation tait.

1. **Sur la référence** (période où les labels sont connus), calculer la perte par observation :
   `abs(y_true - y_pred)` pour la MAE, `np.square(y_true - y_pred)` pour la RMSE.
2. **Entraîner un nanny model** `h` qui prédit cette perte à partir de
   `feature_column_names + [y_pred]` : les features du modèle surveillé **plus sa prédiction**.
   Par défaut, `h` est un `LGBMRegressor()` **sans aucun hyperparamètre** (100 arbres, 31
   feuilles, perte L2), un modèle par métrique. Le tuning FLAML est optionnel et désactivé.
3. **Sur l'analyse** (sans labels), prédire la perte ligne par ligne, la tronquer à 0, puis
   agréger par chunk : moyenne pour la MAE, racine de la moyenne pour la RMSE.

Ce qui en découle, et que la documentation ne dit pas :

- **Le nanny model est ajusté et évalué sur la même référence.** Les seuils d'alerte sont la
  moyenne ± 3 σ de la performance *réalisée* des chunks de référence (`_alert_thresholds`), mais
  les estimations de référence affichées sont en échantillon, donc flatteuses. Je mesure ce biais
  avec une variante cross-fittée (rapport, E1).
- **Le typage des colonnes se fait par dtype.** `nannyml.base._split_features_by_type` : une
  colonne est catégorielle si son dtype est `object`, `string`, `category` ou `bool`. Une heure ou
  un jour de semaine codés en `int` deviennent **continus**, sans avertissement, pour DLE et pour
  la PCA (piège P1 des notes d'implémentation).
- **Télémétrie.** La librairie envoie des événements d'usage à `api.segment.io` par défaut ;
  `NML_DISABLE_USAGE_LOGGING=1` les coupe (piège P2).
- **Bande de confiance ≠ seuil d'alerte.** La bande est estimation ± 3 erreurs-types, l'erreur-type
  étant l'écart-type des pertes de référence divisé par √(taille du chunk). Seul le seuil
  déclenche une alerte. La documentation reconnaît que cette erreur-type suppose des
  observations i.i.d., ce que des heures consécutives ne sont pas.

### 1.2 Les hypothèses, telles que NannyML les écrit

Citées verbatim (section *Assumptions and limitations* de DLE) :

1. « **There is no concept drift.** While dealing well with covariate shift, DLE will not work
   under concept drift. »
2. « **There is no covariate shift to previously unseen regions in the input space.** […] The same
   applies to the nanny model — it cannot predict how big of an error the monitored model will
   make. »
3. « The noise is heteroscedastic around the monitored model target and it is dependent on the
   monitored model input features. » (sinon DLE fonctionne, mais la performance vraie est
   constante en MAE / MSE)
4. « The sample of data used for estimation is large enough. »

Le brief formule l'insight du jour comme une découverte (« capture le covariate shift mais ment
sous concept drift pur »). C'est en fait **l'hypothèse n° 1 de la librairie**. L'apport du projet
est ailleurs : mesurer *où* et *combien* elle ment sur un vrai choc, et montrer que l'hypothèse
n° 2 (support) est aussi décisive que la n° 1.

### 1.3 Pourquoi le forecasting est un cas particulier

DLE vient du monde tabulaire, où une ligne est indépendante des autres. En forecasting
autorégressif, trois choses changent, et aucune n'est discutée dans la documentation :

- **Les features sont des labels passés.** Avec des lags ≥ H, la feature `lag_168` d'une cible t
  est le label de t − 168. Un choc né dans la cible n'apparaît dans les entrées qu'après H heures.
  Pendant cette fenêtre, il est vu comme un concept drift pur (hypothèse H-AR de la SPEC).
- **Le délai des labels est structurel.** La vérité de t arrive à t, donc H après l'émission.
  Mais en temps d'horloge, la performance réalisée des cibles récentes est disponible tout de
  suite : le gain de DLE n'est pas « H jours d'avance » en général (hypothèse H-horloge).
- **Le support se déplace avec le niveau.** Un modèle sur niveaux absolus sort du support à la
  première baisse durable de la demande. Un modèle normalisé (D1) reste dans le support de ses
  entrées mais pas dans celui de sa prédiction, qui est une entrée du nanny model.

### 1.4 Dérive univariée et multivariée : les défauts réels

| Méthode | Défaut de NannyML 0.13.1 (calculateur) |
|---|---|
| Méthodes par défaut | Jensen-Shannon, pour les colonnes continues comme catégorielles |
| KS, JS, Wasserstein, Hellinger, L∞ | seuil supérieur seul : moyenne + 3 σ de la statistique sur les chunks de référence |
| χ² | alerte si p < 0,05, seuil non modifiable |
| PCA | 65 % de variance, seuil moyenne ± 3 σ de l'erreur de reconstruction de référence |
| KS et Wasserstein | calcul exact si la référence a moins de 10 000 lignes (ici 8 760), sinon par quantiles / bacs |

La documentation se contredit : les docstrings des méthodes annoncent encore « KS alerte si
p < 0,05 » et « JS alerte si distance > 0,1 », alors que le dictionnaire de seuils du calculateur
applique moyenne + 3 σ. C'est le calculateur qui s'applique (vérifié dans
`nannyml/drift/univariate/calculator.py`, `DEFAULT_THRESHOLDS`).

---

## 2. DriftGuard (Alam, Rahman, Sadeqa — arXiv:2601.08928, v1, 13 janvier 2026)

### 2.1 Ce que propose le papier

Cinq modules : prévision de base (Prophet, XGBoost, LSTM ; WMAPE 0,048 à 0,052), injection de
dérive, **ensemble de quatre détecteurs** (ratio d'erreur RMSE récent / RMSE de base, KS et PSI
sur les features, autoencodeur sur les résidus, CUSUM sur les résidus ; dérive signalée si au
moins trois sur quatre votent), diagnostic SHAP agrégé du SKU au magasin puis à la région (c'est
le sens de « hiérarchique »), et réentraînement sélectif sous contrainte de coût (fenêtre choisie
dans {30, 60, 90, 180} jours, seuls les 15-20 % de SKU les plus touchés).

### 2.2 Ce que valent les chiffres

L'abstract dit, verbatim : « 97.8% detection recall within 4.2 days ». La section V.A précise
« mean latency » et un taux de faux positifs de 3,2 %. Mais :

- la section III.C annonce 98 % de rappel, 3 % de faux positifs et une latence **médiane** de
  5 jours ; la figure 1 dit « 3-day Latency » ; le tableau III donne 5,1 et 4,2 jours sans
  expliquer la seconde colonne ;
- la précision est définie mais jamais chiffrée ; le dénominateur des faux positifs n'est pas
  défini ; ni seuils, ni fenêtres, ni graines, ni nombre de répétitions ;
- le tableau I porte sur « 5 040 SKU dans 10 magasins », pas sur les 30 490 séries annoncées ;
- la figure 3 montre des magasins de New York et de Floride, absents de M5 (Californie, Texas,
  Wisconsin, comme le papier le dit lui-même) ;
- la dérive est injectée au jour 1 800 pour 113 jours, soit jusqu'au jour 1 913 : **dans** la
  fenêtre d'entraînement et de validation (jours 1-1 913), pas dans la fenêtre de test ;
- aucune version publiée ni révision par les pairs n'est indiquée, et aucun code n'est fourni.

Conclusion de veille : le papier est utile pour son **cadre d'évaluation** (dérive injectée à une
date connue, rappel × délai × faux positifs), pas pour ses chiffres, qu'on ne peut ni reproduire
ni réconcilier entre eux.

### 2.3 Ce que j'en retiens pour W47

1. **La métrique d'un système de monitoring est un triplet**, pas un couple : rappel, délai *et*
   fausses alertes. Le brief retient « rappel × délai » ; sans les fausses alertes, un détecteur
   qui alerte tous les jours a 100 % de rappel et un délai nul.
2. **Le délai doit dire de quelle horloge il parle.** DriftGuard le définit comme « days between
   drift onset and detection » sans préciser si la détection utilise des labels. Un détecteur à
   base d'erreur (le ratio RMSE) a besoin des labels, donc il paie leur latence. Le banc E3 donne
   les deux axes.
3. **Les dérives de DriftGuard changent la cible** (saisonnalité × (1 − 0,4), niveau × 0,8,
   volatilité × 3…). Ce sont des concept drifts au sens de NannyML, donc exactement le cas où DLE
   se déclare impuissant. Un ensemble qui inclut un détecteur fondé sur l'erreur réalisée n'est
   pas « sans labels ». Les deux outils ne répondent pas à la même question.
4. **Le vote 3 sur 4** est une version de la « condition jointe » du bloc 3 ; elle réduit les
   fausses alertes au prix du délai, ce que E3 mesure pour la règle « dérive ET DLE ».

---
