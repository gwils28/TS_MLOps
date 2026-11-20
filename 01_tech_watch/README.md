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
