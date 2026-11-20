# Plan W47 (brief d'origine, reproduit tel quel)

> Ce fichier conserve le brief reçu le 20 novembre 2026 (semaine ISO 47), sans modification de fond,
> pour que chaque écart soit traçable. Les liens entre crochets sont ceux du brief ; leur
> vérification est faite dans [`01_tech_watch/README.md`](01_tech_watch/README.md#verification-des-citations-du-brief)
> (plusieurs ne soutiennent pas l'affirmation à laquelle ils sont attachés).

**Thème du jour :** MLOps & observabilité des modèles — surveiller un modèle de forecasting en
production, où la vérité terrain arrive après l'horizon de prévision. Détecter la dégradation
silencieuse avant qu'elle ne coûte de l'argent.

**Le réflexe du jour :** un modèle ne « tombe » pas en panne, il dérive en silence. La compétence
senior, c'est de mesurer sa santé sans attendre les labels — et de savoir où cette mesure devient
un mensonge.

## Livrable 1 — Plan d'action du jour (~55 min)

### 1. Veille technologique & scientifique (12 min)

- **NannyML — post-deployment data science (lib open-source)** → https://github.com/nannyml/nannyml ·
  doc : https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html
  *Pertinent aujourd'hui :* le cœur du problème forecasting. Quand le résultat réel de tes
  prédictions est retardé, ou que les labels sont totalement absents, NannyML estime la
  performance via CBPE (classification) ou DLE (régression) — n'importe quelle métrique, ex. RMSE.
  Pour la dérive multivariée, NannyML utilise une reconstruction de données par PCA et surveille
  la dérive de l'erreur de reconstruction. Le DLE est ton outil : il entraîne un « nanny model »
  pour estimer la perte du modèle surveillé observation par observation.
  [QWE AI Academy](https://www.qwe.edu.pl/tutorial/claude-desktop-linux-unofficial-guide/)
- **DriftGuard: A Hierarchical Framework for Concept Drift Detection and Remediation in Supply
  Chain Forecasting** — Alam et al., arXiv:2601.08928, janvier 2026 → https://arxiv.org/abs/2601.08928
  *Pertinent aujourd'hui :* dans le thème et le secteur du jour. Les modèles de forecasting
  supply-chain se dégradent à mesure que les conditions réelles changent — promotions,
  préférences, ruptures — provoquant une dégradation silencieuse qui mène à des ruptures de stock
  ou des surstocks sans déclencher la moindre alerte système. Côté chiffres, évalué sur plus de
  30 000 séries du jeu M5, DriftGuard atteint 97,8 % de rappel de détection en 4,2 jours — note
  bien le couple recall × délai de détection, la vraie métrique d'un système de monitoring.
  [arxiv](https://arxiv.org/pdf/1808.08778)

### 2. Ingénierie & code (25 min)

**Objectif :** détecter une dégradation avant l'arrivée des labels, puis vérifier si ton
estimateur disait vrai.

**Consigne précise :**

1. Prends une série de demande réelle (ex. trajets NYC TLC agrégés par heure, cf. Livrable 2).
   Entraîne un forecaster de régression (LightGBM avec lags/calendrier) sur une période de
   référence.
2. Génère les prédictions sur une période d'analyse contenant un choc connu (la chute COVID 2020,
   ou un shift injecté artificiellement).
3. Avec NannyML DLE, estime le RMSE/MAE sur la période d'analyse sans utiliser les vraies
   valeurs ; lance en parallèle la dérive univariée (KS/Wasserstein) et multivariée
   (reconstruction PCA).
4. Révèle ensuite les vraies cibles et superpose erreur estimée vs erreur réalisée.

**Stack :** Python — `nannyml`, `lightgbm`/`scikit-learn`, `pandas`.

**Critère de réussite — tu sais que c'est fini quand :** ta courbe d'erreur estimée suit l'erreur
réalisée à travers le choc covariable, et tu peux nommer le régime où l'estimation décroche.
L'insight senior : l'estimation de performance sans labels capture le covariate shift mais ment
sous concept drift pur (la relation X→y a changé, l'estimateur ne le voit pas). Savoir tracer
cette frontière, c'est le livrable du jour.

### 3. System Design & MLOps (15 min)

Mini-décision d'architecture : conçois l'observabilité d'un modèle de forecasting dont la vérité
terrain arrive avec H jours de retard (l'horizon). Produis un raisonnement structuré (pas de code)
sur :

- **Indicateurs avancés (avant labels).** Que surveilles-tu pendant la fenêtre aveugle ? Dérive des
  entrées (univariée + multivariée PCA), dérive des prédictions, et performance estimée (DLE).
  Comment les hiérarchises-tu ?
- **Réduire le bruit d'alerte.** La bonne pratique 2026 : ne pas alerter sur la dérive (PSI) seule,
  mais sur la condition jointe « dérive des entrées + chute du score d'évaluation » — la dérive
  sans impact mesurable est une fausse alerte qui épuise l'astreinte. Comment encodes-tu ce ET
  logique ? [Substack](https://aiblewmymind.substack.com/p/claude-cowork-scheduled-tasks-6-ways)
- **Politique de réentraînement.** Ne pas réentraîner par réflexe : seulement si la dérive est
  confirmée ET que tu disposes de suffisamment de nouvelles données labellisées — réentraîner sur
  trop peu de données peut empirer les choses. Quel est ton garde-fou contre les boucles de
  rétroaction (le modèle qui influence la demande qu'il prédit) ?
  [Artificialcorner](https://artificialcorner.com/p/claude-automates)

8-12 lignes ou schéma textuel.

### 4. Lecture de référence

Sculley et al. (2015), *Hidden Technical Debt in Machine Learning Systems* (NeurIPS 28) →
https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html

*Ce que tu en retires :* le papier fondateur du MLOps. Il montre qu'il est dangereux de croire les
gains rapides « gratuits » : les systèmes ML réels accumulent des coûts de maintenance massifs, via
l'érosion des frontières, l'entanglement, les boucles de rétroaction cachées, les dépendances de
données, et les changements du monde extérieur. Ce dernier point — *changes in the external world*
— est exactement le drift que tu monitores aujourd'hui. À lire comme la grille de lecture de toute
ta pratique d'observabilité. [arxiv](https://arxiv.org/pdf/2601.03099)

**Récap temps :** 12 + 25 + 15 + ~5 ≈ 57 min. Si tu débordes, garde la dérive multivariée du bloc 2
pour demain ; l'essentiel est le décrochage estimé-vs-réalisé.

## Décisions prises en interview (20/11/2026)

| Question | Réponse |
|---|---|
| Dépôt | `gwils28/TS_MLOps` |
| Format | Site Quarto (`report/` → `docs/`, GitHub Pages), même structure que W25 |
| Langue | Rapport et documentation en français, code et docstrings en anglais |
| Scénarios | Choc réel COVID 2020 · concept drift pur injecté · covariate shift pur injecté · banc recall × délai à la DriftGuard |
| Git | Claude commite, pousse et fusionne (dérogation explicite à l'article de méthode) |
