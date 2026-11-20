# Source verification notes (retrieved 2026-11-20)

All files downloaded to `raw/` next to this file. Quotes are verbatim from the retrieved text; PDF text was extracted with pypdf, which inserts spurious spaces inside some words (e.g. "dan gerous") and drops ligatures; those artifacts have been silently repaired in the quotes below, the wording is otherwise unchanged. Page numbers = PDF page index (the printed page number is identical for both PDFs).

---------------------------------------------------------------------
## 1. DriftGuard (arXiv:2601.08928)

- Abs page: https://arxiv.org/abs/2601.08928 (HTTP 200) ; PDF: https://arxiv.org/pdf/2601.08928 (HTTP 200, 8 pages) -> raw/abs_driftguard.html, raw/driftguard.pdf, raw/driftguard.txt
- Title: "DriftGuard: A Hierarchical Framework for Concept Drift Detection and Remediation in Supply Chain Forecasting"
- Authors: Shahnawaz Alam, Mohammed Abdul Rahman (Muffakham Jah College of Engineering and Technology, Hyderabad), Bareera Sadeqa (Shadan Women's College of Engineering and Technology, Hyderabad).
- Version: only v1, "Tue, 13 Jan 2026 19:08:11 UTC (18 KB)". Subjects: cs.LG; cs.CV (sic). No journal ref / DOI / comments field. Not peer-reviewed as far as the abs page shows. No code or data release mentioned anywhere in the PDF.

### Headline claim (verbatim)
- Abstract (p.1): "Evaluated on over 30,000 time series from the M5 retail dataset, DriftGuard achieves 97.8% detection recall within 4.2 days and delivers up to 417× return on investment through targeted remediation." -> "97.8% detection recall within 4.2 days" IS verbatim. (On the HTML abs page the "×" is lost: "up to 417 return on investment".)
- Sec. V.A (p.7): "Our ensemble detection approach achieves 97.8% recall with a mean latency of 4.2 days—substantially outperforming any individual detector operating alone."
- Conclusion (p.8): "97.8% detection recall at 4.2-day mean latency".
- False positives (p.7): "Importantly, the false positive rate of 3.2% means that unnecessary retraining is triggered approximately once per month—an acceptable overhead given the cost of missed drift events. In production deployments, this rate could be further reduced through domain-specific threshold tuning."
- PRECISION: defined in Sec. IV.C but NO numeric precision value is reported anywhere.

### Internal inconsistencies in the numbers (important)
- Sec. III.C (p.4): "Performance: 98% recall, 3% FPR, 5-day median latency [Fig. 2]." (vs 97.8% / 3.2% / 4.2-day mean in Sec. V).
- Fig. 1 (p.4) labels: "Ensemble Detection (98% recall)", "3-day Latency", "16% Recovery", "Drift Injection (-25% acc)", "Baseline Forecasting (95% acc)".
- Table III (p.7) "End-to-End Pipeline Performance": row "Latency (days) | - | 5.1 | 4.2 | -" under columns Baseline / Post-Drift / Retrained / Recovery; row "WMAPE 0.048 | 0.192 | 0.057 | +13.5%". Latency attached to a "Retrained" column is not explained; "+13.5%" recovery vs "16% Recovery" in Fig. 1 not reconciled.
- So the paper gives four latency figures: 3 days (Fig.1), 4.2 days mean (abstract, V.A, VII, Fig.2), 5-day median (III.C), 5.1 (Table III).
- Dataset size: abstract/IV.A say M5 30,490 series; but Sec. III.A (p.4) says Table I baselines were evaluated "on our evaluation dataset comprising 5,040 SKUs across 10 retail stores over 1,969 days". Which subset the detection numbers use is not stated explicitly.
- Fig. 3 (p.6) shows "CA Store 23% drop, TX Store 12% drop, NY Store 8% drop, FL Store 3% drop", but M5 has stores only in California, Texas, Wisconsin (the paper itself says so in IV.A). NY/FL do not exist in M5.
- Train/test window: "Following competition protocols, we use days 1-1913 for training and validation, reserving days 1914-1941 for testing." vs "We inject drift starting at day 1800, providing 113 days of post-drift observations" (1800+113 = 1913), i.e. the drift window lies inside the training/validation span, not the test span. Not explained.
- Reference [17] cites "D. Sculley et al., 'Machine learning: The high-interest credit card of technical debt,' in Proc. NeurIPS Workshop ML Systems, 2022." That paper is the 2014 NIPS SE4ML workshop paper (Sculley 2015 itself lists it as ref [2], "SE4ML: Software Engineering for Machine Learning (NIPS 2014 Workshop)"). Citation year/venue wrong.
- The 2020 grocery "toilet paper" anecdote (p.1) is unsourced.

### Architecture ("five-module framework")
Fig. 1 / Sec. III: (1) Baseline Forecasting Engine, (2) Drift Scenario Generation (injection), (3) Multi-Layer Ensemble Drift Detection, (4) Intelligent Drift Diagnosis (SHAP), (5) Cost-Aware Adaptive Retraining. Conclusion lists them differently: "baseline establishment, ensemble detection, SHAP-based diagnosis, hierarchical impact assessment, and cost-aware adaptive retraining."
- Baseline models (Table I, p.4): Prophet WMAPE 0.052, XGBoost 0.048, LSTM 0.051. XGBoost features: lags y_{t-1}, y_{t-7}, y_{t-28}, y_{t-364}; calendar; price; 7/28-day rolling mean/std. Bottom-up reconciliation: "forecasts at aggregate levels (store totals, category subtotals) are computed by summing constituent SKU-level predictions."
- Hierarchy (Sec. IV.A, p.6): "organized in a three-level hierarchy: Geographic: 10 stores across 3 states (California, Texas, Wisconsin); Product: 3,049 products across 3 categories (Foods, Household, Hobbies) and 7 departments; Temporal: Daily granularity with weekly and monthly aggregation levels".
- Detection (Sec. III.C, p.4; Algorithm 1, p.5): four detectors, binary votes, "drift is signaled when at least three of four detectors agree—a threshold chosen to balance sensitivity against false positive rates."
  1. Error-based: e1 = RMSE_recent / RMSE_baseline − 1 (Eq. 2); "may miss gradual shifts or compensating errors".
  2. Statistical feature monitoring: "Kolmogorov-Smirnov test for continuous features and Population Stability Index (PSI) for categorical variables" (Algorithm 1 comment says "{KS, PSI, ADF}").
  3. Autoencoder on forecast residuals: e3 = ||r_t − r̂_t||², r_t = y_t − ŷ_t.
  4. CUSUM on residuals: C_t = max(0, C_{t−1} + r_t − µ0 − k).
  Thresholds θe, θs, θa, θc, window lengths, µ0, k: NOT given numerically.
- "Hierarchical" part = hierarchical propagation/impact analysis: SHAP Δφ_i = mean φ_i(drift) − mean φ_i(baseline) (Eq. 7) "aggregating these shifts across the hierarchy—from individual SKUs to stores to regions—we construct a diagnostic map showing drift severity at each level."

### Ground truth / evaluation protocol
- Semi-synthetic injection (Sec. III.B, p.4): "Since naturally occurring drift in production data lacks ground truth labels and timing information, we adopt a semi-synthetic approach: applying realistic transformations to historical M5 data at known time points, enabling precise measurement of detection latency and accuracy."
- Five scenarios (Table II + text): Seasonality Shift y'=y·(1−0.4·s_t) (95.2%→87.4% "accuracy"); Trend Change y'=y−β(t−t0) (94.8%→78.2%; β not given); Level Shock y'=0.8·y (95.5%→68.3%); Volatility Spike y'=ȳ+3(y−ȳ) (94.9%→72.1%); Hierarchical Drift = region-specific transformations (95.1%→81.6%). "Accuracy" metric undefined (presumably 1−WMAPE; not stated).
- Injection protocol (Sec. IV.B, p.6): "We inject drift starting at day 1800, providing 113 days of post-drift observations for detection and response evaluation. Each drift scenario (Table II) is applied to a random 20% subset of SKUs".
- Metrics (Sec. IV.C, p.6): "Recall (proportion of true drift events detected), precision (proportion of detected events that are actual drift), and latency (days between drift onset and detection)." WMAPE = Σ|y−ŷ| / Σy. Business impact via newsvendor, holding cost ratio 0.3, stockout cost ratio 0.7.
- NOT specified: what one "drift event" is (per SKU? per scenario? per hierarchy node?), how many events, how undetected events enter the mean latency, the FPR denominator (per series-day? per window? on the 80% non-drifted SKUs?), number of repetitions/seeds, confidence intervals. ("approximately once per month" for 3.2% FPR is consistent with a per-day rate, but that is my inference, not stated.)
- Implementation: Python 3.9, scikit-learn 1.0, XGBoost 1.5, SHAP 0.40, PyTorch 1.10; AWS c5.4xlarge; ~72 h total runtime.

### Baselines
- Only the four individual component detectors (Fig. 2 "Detection latency-recall tradeoff across methods": Error-based, Autoencoder, Statistical, CUSUM, Ensemble). Text (p.7): "The error-based detector achieves 92% recall but requires 12 days to confidently distinguish drift from normal variance. Statistical tests detect faster (6 days) but produce more false positives. The autoencoder approach shows intermediate characteristics." No numbers for CUSUM in text; no table of per-detector recall/latency/FPR.
- ADWIN, DDM, Page-Hinkley, KSWIN are discussed in related work but NOT compared experimentally. Fixed-schedule retraining is criticised but not quantified as a baseline.

### Remediation (Sec. III.E, p.6)
- Training window: w* = argmin_{w∈{30,60,90,180} days} [C_compute(w) + λ·L_val(w)] (Eq. 8); λ not given.
- Selective updating: S_retrain = TopK({s : ΔWMAPE_s > τ}) (Eq. 9); K, τ not given. Sec. V.C: "rather than retraining all 30,000+ models, we update only the 15-20% most severely affected".
- ROI = (ΔC_inventory − C_compute)/C_compute (Eq. 10). $10.2M → $14.3M → $10.6M inventory cost; compute ~$9,600/yr; "up to 417×". (Arithmetic check, mine: (14.3−10.6)M/9.6K ≈ 385, (4.1M−9.6K)/9.6K ≈ 426; neither equals 417 exactly. Not verifiable from paper.)
- Severe drift: "Severe drift (>30% shift) ... accuracy degrades to 0.217 WMAPE and recovery only reaches 0.098"; "our framework appropriately triggers alerts for human review and potential model architecture changes rather than continuing fully automated remediation."

### Stated limitations (Sec. V.B, VI)
- Incomplete recovery under severe drift ("a fundamental limitation").
- Cold start: "requires a stable baseline period to calibrate detection thresholds and train the autoencoder component."
- Scalability: "Enterprise deployments with millions of SKU-location combinations would require additional engineering ... SHAP-based diagnosis introduces computational overhead that grows with feature dimensionality."
- Human integration; future work: multivariate dependencies (substitution, supply constraints), streaming detection "could reduce latency below the current 4.2-day average", federated detection.

### Notes for replicating a "recall × delay" evaluation
- Ground truth onset is known by construction (day 1800); latency = detection day − onset day.
- Missing for replication: event definition, detection granularity (series vs aggregate), how detections before onset count (false alarms), FPR denominator, detector thresholds/windows, which 20% SKUs (random seed), number of runs, whether evaluation is on 30,490 series or 5,040 SKUs. Recall/latency/FPR figures are internally inconsistent across sections (see above).
- Injected drift alters y only (y'_t); with lag features this also shifts X downstream. These scenarios are label/concept-type changes – note for NannyML DLE, which explicitly assumes no concept drift (see §2).

---------------------------------------------------------------------
## 2. NannyML documentation ("stable" = NannyML 0.13.1 documentation, per page titles)

Pages retrieved (all HTTP 200):
- PE: https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html
- DLE tutorial: https://nannyml.readthedocs.io/en/stable/tutorials/performance_estimation/regression_performance_estimation.html
- Multivariate: https://nannyml.readthedocs.io/en/stable/how_it_works/multivariate_drift.html (#data-reconstruction-with-pca)
- PCA tutorial: https://nannyml.readthedocs.io/en/stable/tutorials/detecting_data_drift/multivariate_drift_detection/pca.html
- Univariate methods: https://nannyml.readthedocs.io/en/stable/how_it_works/univariate_drift_detection.html
- Univariate tutorial: https://nannyml.readthedocs.io/en/stable/tutorials/detecting_data_drift/univariate_drift_detection.html
- Thresholds (how it works): https://nannyml.readthedocs.io/en/stable/how_it_works/thresholds.html
- Thresholds tutorial: https://nannyml.readthedocs.io/en/stable/tutorials/thresholds.html
- Sampling error: https://nannyml.readthedocs.io/en/stable/how_it_works/estimation_of_standard_error.html
- API: .../nannyml/nannyml.performance_estimation.direct_loss_estimation.dle.html, .../nannyml/nannyml.drift.multivariate.data_reconstruction.calculator.html, .../nannyml/nannyml.drift.univariate.calculator.html, .../nannyml/nannyml.drift.univariate.methods.html

### CBPE vs DLE scope
- PE page intro: "CBPE is used to estimate performance of classification models as they return predictions with an associated confidence score." ... "The idea behind DLE is to train a nanny model that directly estimates the value of the loss of the monitored model for each observation (sometimes referred to as a child model). This approach is used to estimate the performance of regression models as the value of the loss function can be calculated for a single observation and turned into performance metrics (unlike classification metrics - one cannot calculate F1 for a single prediction)."
- CBPE: "Assuming properly calibrated probabilities, confusion matrix elements can be estimated and then used to calculate any performance metric." -> "any metric" applies to confusion-matrix classification metrics, not RMSE.
- DLE supported metrics (tutorial + API): mae, mape, mse, rmse, msle, rmsle. API: "metrics ... default = ['mae', 'mape', 'mse', 'rmse', 'msle', 'rmsle']".

### DLE mechanics (PE page, "Implementation details")
- "For each observation of reference data calculate loss which in case of MAE is absolute error of f, i.e. AE_reference = |y_reference − f(X_reference)|."
- "Train a nanny model on reference data. As features use the monitored model features X_reference and monitored model predictions f(X_reference). The target is absolute error AE_reference calculated in previous step. So ÂE = h(X, f(X))."
- "Estimate performance of child model on analysis data. Estimate absolute error for each observation ... with h and, finally calculate the mean ... to get MAE." (the page writes \hat{AE}_{reference} here; it means analysis data – typo in docs)
- "For other metrics step 1 and 3 are slightly modified. For example, for root mean squared error (RMSE) in step 1 we would calculate the squared error while in step 3 we would calculate the root of the mean of all predictions."
- Loss is "directionless": "DLE tries to predict loss which is directionless (like already mentioned absolute or squared error). This is a significantly easier problem to solve. For this reason, the nanny model algorithm does not have to be better than the child algorithm."
- "The nanny model uses LGBM [5] algorithm." "The nanny model is trained on the reference dataset." "The child model prediction is used as an input feature for the nanny model." Hyperparameter tuning optional via FLAML. "By default LGBM uses squared error (L2) metric. Absolute error (L1) is worth consideration when the user expects more stable loss estimation".
- API (DLE class): "It uses an internal LGBMRegressor model per metric to predict the value of the error function ... of the monitored model. The error results on the reference data become a target for those internal models." tune_hyperparameters default False; default FLAML config time_budget 15, metric "mse", n_splits 5, etc.
- Exact per-observation loss for msle/rmsle/mape is not spelled out on the how-it-works page (only MAE and RMSE examples).

### DLE: confidence band, sampling error, thresholds, alerts
- Tutorial: "upper_confidence_boundary and lower_confidence_boundary - These values show the Confidence Band of the relevant metric and are equal to estimated value +/- 3 times the estimated Sampling Error."
- Sampling-error page: "Since targets are available only in the reference dataset, the nominator of the SEM formula is calculated based on observation-level metrics from the reference dataset. The sample size in the denominator is the size of the chunk for which standard error is estimated." ... "The sampling error in the results is expressed as 3 standard errors. So the estimated performance +/- 3 standard errors creates an interval which should contain the actual value of performance metric in about 99% of cases (given the assumptions of the performance estimation algorithm are met)."
  (Check in tutorial output: RMSE value 1067.21, sampling_error 10.348, upper CB 1098.26 = 1067.21 + 3×10.348.)
- Limitations of SE: "the standard error is based on reference data. The only information it takes from the analysis chunk is its size. Therefore, it provides accurate estimations for the analysis period as long as the i.i.d (independent and identically distributed) assumption holds." Also: "In many cases chunks are not independent either e.g. when observations in chunks are selected chronologically, not randomly."
- Thresholds (tutorial): "The thresholds are calculated based on the actual performance of the monitored model on chunks in the reference partition. By default, the thresholds are 3 standard deviations away from the mean performance calculated on chunks. They are calculated during fit phase." Alert: "True if estimated performance crosses upper or lower threshold." API default: every DLE metric → StandardDeviationThreshold() (multipliers 3/3, offset from mean).
- Note: confidence band (±3 SE, sampling error) and alert thresholds (mean ± 3 std of realized reference-chunk metric) are two different things; alerts are driven by thresholds only.

### DLE "Assumptions and limitations" (verbatim)
- "In general, DLE works well if there are regions in the feature space where the model performs better or worse and there are enough observations from these regions in the reference dataset so that the nanny model can learn this pattern. Just like CBPE, it will handle covariate shifts well."
- "There is no concept drift. While dealing well with covariate shift, DLE will not work under concept drift. This shouldn't happen when the child model has access to all the variables affecting the outcome and the problem is stationary." (example: heating energy demand; prices rising → concept drift)
- "There is no covariate shift to previously unseen regions in the input space. The monitored model will most likely not work if the drift happens to subregions in the inputs space that were not seen before. ... The same applies to the nanny model - it cannot predict how big of an error the monitored model will make."
- "The noise is heteroscedastic around the monitored model target and it is dependent on the monitored model input features. This is equivalent to there are regions where the monitored model performs better or worse. DLE also works when the noise is homoscedastic ... but then the true performance of the monitored model is constant (depending on the metric used, it will be constant for MAE and MSE, it will change when measured e.g. with MAPE)."
- "The sample of data used for estimation is large enough. When the sample size is small, the actual performance calculated on the sample is not reliable as it is a subject of random sampling effects (sampling error)."
- CBPE equivalent: "There is no concept drift. While dealing well with data drift, CBPE will not work under concept drift i.e. when P(Y|X) changes. Except from very specific cases, there is no way to identify concept drift without any ground truth data."

### Multivariate drift: Data Reconstruction with PCA (how-it-works)
- "The first step is data preparation. This includes missing values Imputation, frequency encoding, and scaling the data. Missing values need to be imputed because it is a PCA requirement. Frequency encoding is used to convert all categorical features into numbers. The next thing to do is standardize all features to 0 mean and unit variance".
- "By default it aims to capture 65% of the dataset's variance, but this is a parameter that can be changed. The PCA algorithm is fitted on the reference dataset".
- "The third step is decompressing the data ... using the inverse PCA transformation ... Then, the euclidean distance between the original data points and their re-constructed counterparts is computed. The resulting distances are then aggregated to get their average. The resulting number is called Reconstruction Error."
- Threshold: "NannyML computes the mean and standard deviation of the reconstruction error with PCA on the reference dataset based on the different results for each Data Chunk. ... A threshold for significant change in NannyML is defined as values that are more than three standard deviations away from the mean of the reference data."
- Defaults (API + PCA tutorial): n_components default=0.65 (passed to sklearn PCA); imputer_categorical default most_frequent; imputer_continuous default mean; threshold default StandardDeviationThreshold.
- Sampling error: SE = std of per-observation reconstruction error on reference / sqrt(chunk size); band ±3 SE.
- Same page also documents a second multivariate method, the Domain Classifier (LightGBM, cross-validated AUROC).

### Univariate drift (how-it-works + API)
- KS: "The test statistic is the maximum distance of the cumulative distribution functions (CDF) of the two samples." Exact (raw reference) vs quantile-binned computation: "By default, NannyML employs the first method if the reference data has fewer than 10,000 rows". Same rule for Wasserstein (equal-width bins otherwise).
- JS (continuous): binning with "Doane's formula from numpy"; low-cardinality (<10% of reference size, max 50 unique values) → each value is a bin; out-of-range chunk data → new bin with reference frequency 0.
- Chi2: contingency table; "not recommended for categorical features with many low-frequency categories or high cardinality features, unless the sample size is really large."
- "we drop all the missing values encountered."
- Default methods (API UnivariateDriftCalculator): categorical_methods default=['jensen_shannon']; continuous_methods default=['jensen_shannon'].
- Default thresholds (API UnivariateDriftCalculator): kolmogorov_smirnov, jensen_shannon, wasserstein, hellinger, l_infinity: StandardDeviationThreshold(std_lower_multiplier=None) → upper-only, mean + 3 std of reference-chunk values. "The chi2 method does not support custom thresholds for now ... its current p-value based implementation." Chi2 docstring: "An alert will be raised for a Chunk if p_value < 0.05."
- DOC INCONSISTENCY: the methods module docstrings still say KS "An alert will be raised for a Chunk if p_value < 0.05" and JS "By default an alert will be raised if distance > 0.1"; L-infinity "distance > 0.1". These contradict the calculator's default threshold dict (std-based). The calculator-level defaults are the ones applied by the calculator per its own docs; the method docstrings look stale. Report should cite the calculator defaults and flag the discrepancy.
- Sampling-error page (also stale): "Currently Univariate Drift Detection ... is based on two-sample statistical tests" (written before distance methods).

### Thresholds (general)
- Thresholds tutorial: "Performance metrics and drift detection methods, and the missing values data quality metric have the following default threshold: nml.thresholds.StandardDeviationThreshold(std_lower_multiplier=3, std_upper_multiplier=3, offset_from=np.mean)". Exception: unseen values → ConstantThreshold(lower=None, upper=0).
- How-it-works: thresholds computed from per-chunk reference values; "When calculating the threshold values during fitting, NannyML will check if the calculated threshold values fall within these limits. If they don't, the breaching threshold value(s) will be overridden by the theoretical limit."

---------------------------------------------------------------------
## 3. Sculley et al. 2015, "Hidden Technical Debt in Machine Learning Systems"

- Abstract page: https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html (200)
- PDF: https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf (200; canonical citation_pdf_url .../paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf), 9 pages.
- Authors: D. Sculley, Gary Holt, Daniel Golovin, Eugene Davydov, Todd Phillips, Dietmar Ebner, Vinay Chaudhary, Michael Young, Jean-François Crespo, Dan Dennison (Google). Advances in Neural Information Processing Systems 28 (NIPS 2015). Official BibTeX has an empty `pages = {}` field, so proceedings page range NOT verifiable from the source; page numbers below are PDF pages 1–9.
- p.8: "A short version of this paper was presented at the SE4ML workshop in 2014 in Montreal, Canada."

### Section list
1 Introduction (p.1) · 2 Complex Models Erode Boundaries (p.2: Entanglement, Correction Cascades, Undeclared Consumers) · 3 Data Dependencies Cost More than Code Dependencies (p.3: Unstable Data Dependencies, Underutilized Data Dependencies, Static Analysis of Data Dependencies [p.4]) · 4 Feedback Loops (p.4: Direct, Hidden) · 5 ML-System Anti-Patterns (p.4–6: Glue Code, Pipeline Jungles, Dead Experimental Codepaths, Abstraction Debt, Common Smells) · 6 Configuration Debt (p.6) · 7 Dealing with Changes in the External World (p.7: Fixed Thresholds in Dynamic Systems, Monitoring and Testing → Prediction Bias, Action Limits, Up-Stream Producers) · 8 Other Areas of ML-related Debt (p.7–8: Data Testing, Reproducibility, Process Management, Cultural) · 9 Conclusions: Measuring Debt and Paying it Off (p.8).

### Quotes
- Abstract (p.1): "Using the software engineering framework of technical debt, we find it is common to incur massive ongoing maintenance costs in real-world ML systems."
- Intro (p.1): "Hidden debt is dangerous because it compounds silently."
- CACE (p.2): "Machine learning systems mix signals together, entangling them and making isolation of improvements impossible. ... If we change the input distribution of values in x1, the importance, weights, or use of the remaining n − 1 features may all change. ... No inputs are ever really independent. We refer to this here as the CACE principle: Changing Anything Changes Everything."
- Entanglement mitigation (p.2): "A second possible strategy is to focus on detecting changes in prediction behavior as they occur. ... Metrics that operate on a slice-by-slice basis may also be extremely useful."
- Undeclared consumers (p.2): "Without access controls, some of these consumers may be undeclared, silently using the output of a given model as an input to another system. In more classical software engineering, these issues are referred to as visibility debt." "...they create a hidden tight coupling of model ma to other parts of the stack." "Furthermore, undeclared consumers may create hidden feedback loops".
- Data dependencies (p.3): "data dependencies in ML systems carry a similar capacity for building debt, but may be more difficult to detect."
- Unstable (p.3): "some input signals are unstable, meaning that they qualitatively or quantitatively change behavior over time." Mitigation: "create a versioned copy of a given signal."
- Underutilized (p.3): "underutilized data dependencies are input signals that provide little incremental modeling benefit. These can make an ML system unnecessarily vulnerable to change, sometimes catastrophically so". Types: Legacy, Bundled, ε-Features, Correlated Features. "Underutilized dependencies can be detected via exhaustive leave-one-feature-out evaluations."
- Feedback loops (p.4): "live ML systems ... often end up influencing their own behavior if they update over time. This leads to a form of analysis debt".
- Direct (p.4): "A model may directly influence the selection of its own future training data."
- Hidden (p.4): "A more difficult case is hidden feedback loops, in which two systems influence each other indirectly through the world." (stock-market example)
- Glue code (p.5): "a mature system might end up being (at most) 5% machine learning code and (at least) 95% glue code".
- Fixed thresholds (p.7): "such thresholds are often manually set. Thus if a model updates on new data, the old manually set threshold may be invalid. Manually updating many thresholds across many models is time-consuming and brittle. One mitigation strategy ... thresholds are learned via simple evaluation on heldout validation data."
- Monitoring (p.7): "Comprehensive live monitoring of system behavior in real time combined with automated response is critical for long-term system reliability. The key question is: what to monitor?"
- Prediction bias (p.7): "In a system that is working as intended, it should usually be the case that the distribution of predicted labels is equal to the distribution of observed labels. ... Slicing prediction bias by various dimensions isolate issues quickly, and can also be used for automated alerting."
- Action limits (p.7): "it can be useful to set and enforce action limits as a sanity check. These limits should be broad enough not to trigger spuriously."
- Up-stream producers (p.7): "These up-stream processes should be thoroughly monitored, tested, and routinely meet a service level objective that takes the downstream ML system needs into account."
- Real-time (p.7): "Because external changes occur in real-time, response must also occur in real-time as well."
- Data testing debt (p.7): "Basic sanity checks are useful, as more sophisticated tests that monitor changes in input distributions."
- Measuring debt (p.8): "Technical debt is a useful metaphor, but it unfortunately does not provide a strict metric that can be tracked over time." Questions: "How easily can an entirely new algorithmic approach be tested at full scale?" / "What is the transitive closure of all data dependencies?" / "How precisely can the impact of a new change to the system be measured?" / "Does improving one model or signal degrade others?" / "How quickly can new members of the team be brought up to speed?"
- p.8: "Research solutions that provide a tiny accuracy benefit at the cost of massive increases in system complexity are rarely wise practice."

---------------------------------------------------------------------
## 4. Citations attached in the brief

- https://arxiv.org/abs/1808.08778 (200): Yi Ding, Panos Toulis, "Dynamical systems theory for causal inference with application to synthetic control methods" (stat.ME; v1 27 Aug 2018, v3 29 Feb 2020). Topic: screening synthetic-control donor units by dynamical (nonlinear time-series) relationship; applications tobacco legislation, Brexit. -> Does NOT support any DriftGuard claim (unrelated; no drift detection, no M5).
- https://arxiv.org/abs/2601.03099 (200): Saeyoung Rho, Cyrus Illick, Samhitha Narasipura, Alberto Abadie, Daniel Hsu, Vishal Misra, "Time-Aware Synthetic Control" (cs.LG, econ.EM, stat.ML; v1 6 Jan 2026). Topic: state-space (Kalman/RTS smoother, EM) synthetic control for causal inference with trends. -> Does NOT support the Sculley "hidden technical debt" reading.
- qwe.edu.pl ".../claude-desktop-linux-unofficial-guide/", aiblewmymind.substack.com ".../claude-cowork-scheduled-tasks-6-ways", artificialcorner.com ".../claude-automates": all three BLOCKED by the egress proxy ("CONNECT tunnel failed, response 403"); full paths were elided in the brief, so I tried https://qwe.edu.pl/claude-desktop-linux-unofficial-guide/, https://aiblewmymind.substack.com/p/claude-cowork-scheduled-tasks-6-ways, https://www.artificialcorner.com/p/claude-automates. Content not read. Judging from slugs only: Claude Desktop on Linux (unofficial guide), Claude Cowork scheduled tasks, and Claude automation — none plausibly about NannyML, DLE, drift detection, DriftGuard or Sculley; they should not be used as sources for those claims.
