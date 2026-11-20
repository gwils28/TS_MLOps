# Bloc 4 — Sculley et al. (2015), *Hidden Technical Debt in Machine Learning Systems*

> D. Sculley, G. Holt, D. Golovin, E. Davydov, T. Phillips, D. Ebner, V. Chaudhary, M. Young,
> J.-F. Crespo, D. Dennison (Google), *Advances in Neural Information Processing Systems 28*
> (NeurIPS 2015). [Résumé](https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) ·
> [PDF](https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf).
> Pages citées = pages du PDF (9 pages). Une version courte avait été présentée à l'atelier
> SE4ML de NIPS 2014 (p. 8). Citations verbatim dans [`../01_tech_watch/source_notes.md`](../01_tech_watch/source_notes.md#3-sculley-et-al-2015-hidden-technical-debt-in-machine-learning-systems).

## La thèse en une phrase

Un système de ML accumule une dette de maintenance **silencieuse** (« Hidden debt is dangerous
because it compounds silently », p. 1), parce que ses frontières sont floues : le comportement
dépend des données autant que du code, et les données changent sans prévenir.

## Plan du papier et ce que chaque section dit à W47

| § | Dette | Ce que le papier dit | Où elle se manifeste dans W47 |
|---|---|---|---|
| 2 | **Entanglement, CACE** (« Changing Anything Changes Everything », p. 2) | Changer la distribution d'une entrée change l'usage de toutes les autres. | La correction D1 (cible relative au niveau) a changé **ce que voient les détecteurs de dérive** : le choc COVID n'est plus visible dans les entrées du modèle, seulement dans ŷ et dans le niveau surveillé à part. Modifier le modèle a modifié le monitoring. |
| 2 | **Undeclared consumers** (p. 2-3) | Des systèmes lisent silencieusement les sorties du modèle et créent un couplage caché. | Le nanny model est un consommateur de ŷ : toute évolution du forecaster (D1 en est une) invalide le nanny model, qui doit être réajusté en même temps. À versionner ensemble. |
| 3 | **Unstable data dependencies** (p. 3) | Un signal amont change de comportement ; parade : en geler une copie versionnée. | Fichiers TLC hachés (SHA-256) dans un manifeste ; un re-téléchargement qui ne reproduit pas le hachage fait échouer le pipeline. Les horodatages naïfs (heure d'été) sont un signal instable documenté. |
| 3 | **Underutilized data dependencies** (p. 3) | Des features à faible apport rendent le système fragile ; à détecter par *leave-one-feature-out*. | `month`, vu deux fois à l'entraînement, servait surtout d'identifiant d'année (v0) ; dans le nanny model, il fait « mémoriser » l'erreur mensuelle de la référence (E1, P0). |
| 4 | **Feedback loops** directes et cachées (p. 4) | Un modèle influence la sélection de ses propres données futures, ou deux systèmes s'influencent via le monde. | Un forecaster de demande taxi qui pilote l'allocation de flotte change l'offre, donc les trajets observés : la demande servie (étiquette) n'est plus la demande latente. Garde-fou au bloc 3. |
| 5 | **Glue code, pipeline jungles** (p. 4-6) | « at most 5% machine learning code » (p. 5). | Le code ML (forecaster + appel NannyML) est minoritaire ; l'essentiel est du timing, du découpage, de l'injection et de l'agrégation. C'est là que sont les bugs trouvés (P1 : typage, P4 : cross-fit). |
| 6 | **Configuration debt** (p. 6) | La configuration est du code et doit être revue et testée. | Tous les nombres vivent dans `configs/protocol.yaml`, figé avant les résultats ; chaque écart est consigné (D1, D2). |
| 7 | **Changes in the external world** (p. 7) | Seuils fixes dans un système dynamique ; surveiller le biais de prédiction, poser des *action limits*, surveiller les producteurs amont. | C'est le sujet de W47. Voir ci-dessous. |
| 8 | Data testing, reproductibilité, processus, culture (p. 7-8) | « Basic sanity checks are useful, as more sophisticated tests that monitor changes in input distributions. » | Tests anti-fuite, tests d'invariants des scénarios, réplique DLE testée contre NannyML. |
| 9 | **Mesurer la dette** (p. 8) | Cinq questions (voir plus bas). | Appliquées au projet en fin de page. |

## § 7, lu de près : ce que le papier recommandait déjà en 2015

1. **Seuils fixes** (*Fixed Thresholds in Dynamic Systems*). « Thus if a model updates on new
   data, the old manually set threshold may be invalid. » Parade proposée : apprendre les seuils
   sur des données de validation. NannyML fait exactement cela (moyenne ± 3 σ sur la référence),
   mais le seuil devient faux dès que le niveau de la série change : pendant le confinement, la
   MAE réalisée tombe à 168 trajets pour un seuil calibré à ~2 500. Aucun détecteur absolu ne
   peut alerter dans ce régime. **Le seuil doit vivre à la même échelle que la métrique**
   (relative si le niveau bouge), ce qui motive D2.
2. **Biais de prédiction** (*Prediction Bias*). « the distribution of predicted labels is equal to
   the distribution of observed labels ». C'est un contrôle **à labels**, mais bon marché et
   découpable par tranche. Il complète DLE, qui est aveugle au concept drift : le biais réalisé
   sur les cibles récentes est le meilleur détecteur de concept drift disponible (E3).
3. **Action limits.** « These limits should be broad enough not to trigger spuriously. » C'est la
   version 2015 du « ne pas alerter sur la dérive seule » du brief.
4. **Producteurs amont** (*Up-Stream Producers*). Surveiller la chaîne qui produit les features.
   Ici, la latence des labels L est une propriété du producteur (TLC publie avec environ deux mois
   de délai) : elle fixe la valeur réelle d'un estimateur sans labels.
5. « Because external changes occur in real-time, response must also occur in real-time as
   well. » Le banc E3 mesure ce temps de réponse, sur l'horloge réelle.

## Ce que le papier ne dit pas (et que W47 montre)

- Sculley et al. traitent le changement du monde comme **une** catégorie. Pour un estimateur
  sans labels, il faut en distinguer au moins trois : covariate shift dans le support (estimable),
  covariate shift hors support (non estimable, le nanny model extrapole), concept drift (non
  estimable sans labels). Le choc COVID les traverse toutes les trois en quelques semaines.
- Le papier ne parle pas d'**horloge**. Dans un système à labels retardés, « détecter vite » n'a
  de sens qu'en précisant quand chaque signal est disponible.

## Les cinq questions de mesure de la dette (p. 8), appliquées à W47

| Question | Réponse pour ce dépôt |
|---|---|
| Peut-on tester une approche entièrement nouvelle à pleine échelle facilement ? | Oui pour le monitoring (un `Monitor` se branche sur n'importe quel frame avec `y_pred`) ; non pour le forecaster, dont les features sont figées par la contrainte d'horizon. |
| Quelle est la fermeture transitive des dépendances de données ? | Un seul producteur (TLC), haché ; calendrier fédéral via `holidays` (versionné par `uv.lock`). |
| Peut-on mesurer précisément l'impact d'un changement ? | Oui sur la référence (garde-fou) ; le banc E3 mesure l'impact d'un changement de règle d'alerte en rappel, délai et fausses alertes. |
| Améliorer un modèle dégrade-t-il les autres ? | Oui : D1 a amélioré le forecaster et rendu les détecteurs de dérive d'entrée aveugles au niveau. C'est le CACE en action. |
| Un nouveau venu se met-il vite à niveau ? | `CLAUDE.md`, `SPEC.md`, notes d'implémentation et rapport ; le point le plus difficile à transmettre est le modèle des trois horloges. |
