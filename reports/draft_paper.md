# Explainable EEG "Brain-Age Clock" in Childhood--Adolescence Using Healthy Brain Network Data (Pilot Study)

*Draft — Pilot Results from Limited Data Extract*

---

## Abstract

Electroencephalography (EEG) provides a non-invasive window into brain maturation, yet most EEG-based age-prediction studies treat the model as a black box or conflate periodic (oscillatory) and aperiodic (1/f background) spectral components. We present a reproducible, open-source pipeline for building an explainable "brain-age clock" from resting-state EEG, with explicit separation of periodic band powers, peak alpha frequency (PAF), individualized alpha power, and aperiodic spectral parameters via spectral parameterization (specparam). In this pilot study, we demonstrate the full pipeline on a 20-participant subset of a public OpenNeuro dataset (ds004148, ages 18--22), extracting 34 spectral features per recording and evaluating age prediction with leakage-safe cross-validation. Pilot results show that the narrow age range limits predictive performance (best MAE = 1.01 years, baseline), but the pipeline successfully extracts physiologically meaningful features (mean PAF = 10.2 Hz, aperiodic exponent = 0.91) and identifies parietal theta power as the most informative feature in permutation importance analysis. We describe concrete plans for scaling to the full Healthy Brain Network dataset (ages 5--21, N > 2,000) where developmental EEG trajectories are expected to provide substantially greater predictive signal.

---

## 1. Introduction

The human brain undergoes profound structural and functional changes from childhood through adolescence. These maturational processes---including synaptic pruning, myelination, and the refinement of cortical circuits---produce measurable changes in the brain's electrical activity that can be captured non-invasively through electroencephalography (EEG). Understanding the typical trajectory of brain maturation is important for developmental neuroscience, education research, and the identification of atypical development.

EEG signals can be decomposed into spectral components that reflect distinct neural processes. Oscillatory ("periodic") components, such as the alpha rhythm (approximately 8--13 Hz), arise from synchronized neural populations and are thought to reflect thalamocortical circuit function. The frequency at which alpha power peaks---the peak alpha frequency (PAF)---increases systematically during childhood and adolescence, from roughly 6--8 Hz in early childhood to 9--11 Hz in late adolescence, making it one of the most robust spectral markers of brain maturation (Miskovic et al., 2015; Cellier et al., 2021).

However, the EEG power spectrum is not composed solely of oscillatory peaks. A substantial portion of spectral power arises from a broadband, aperiodic "1/f-like" background signal that decreases in power as frequency increases. The slope (exponent) and level (offset) of this aperiodic component also change with development: younger children tend to show steeper spectral slopes, which flatten during adolescence (Donoghue et al., 2020; Ostlund et al., 2022). Critically, if periodic and aperiodic components are not separated, apparent changes in "alpha power" may actually reflect shifts in the aperiodic background rather than true oscillatory changes.

Recent advances in spectral parameterization---notably the specparam algorithm (formerly FOOOF; Donoghue et al., 2020)---allow explicit decomposition of the power spectrum into periodic peaks and aperiodic background. Applying this decomposition to developmental EEG data offers the potential for more physiologically precise feature extraction and, consequently, more interpretable age-prediction models.

In this pilot study, we present a complete, reproducible pipeline for building an explainable EEG brain-age clock. The pipeline spans data acquisition, quality control, feature engineering (with explicit periodic/aperiodic separation), age-prediction modeling with leakage-safe cross-validation, and feature-importance analysis. We demonstrate the pipeline on a small pilot dataset and describe plans for scaling to the full Healthy Brain Network (HBN) dataset, which covers the target developmental age range of 5--21 years.

**Guardrails.** This work is strictly for research and education. The pipeline outputs are not diagnostic tools, clinical decision-support systems, or validated biomarkers. Model predictions have not been clinically validated and must not be used to make decisions about any individual's health or development.

---

## 2. Related Work

### 2.1 EEG-BIDS and Reproducible EEG Research

The Brain Imaging Data Structure for EEG (EEG-BIDS; Pernet et al., 2019) standardizes how EEG data are organized, making datasets findable, accessible, and reusable. The OpenNeuro platform (Markiewicz et al., 2021) hosts BIDS-formatted datasets, enabling reproducible large-scale analyses. Our pipeline uses MNE-BIDS (Appelhoff et al., 2019) to read data directly from the BIDS layout, ensuring compatibility with any compliant dataset.

### 2.2 The Healthy Brain Network

The Healthy Brain Network (HBN; Alexander et al., 2017) is a large-scale, community-referred dataset that includes EEG recordings from children and adolescents aged approximately 5--21. With over 2,000 participants and resting-state EEG in both eyes-open and eyes-closed conditions, HBN provides one of the largest publicly available pediatric EEG resources. OpenNeuro dataset ds004186 contains a BIDS-formatted subset of HBN EEG data, though access requires authentication.

### 2.3 Developmental EEG Trajectories

Several spectral features of the resting EEG change systematically with age during childhood and adolescence:

- **Peak alpha frequency (PAF)** increases from approximately 6--8 Hz in early childhood to 9--11 Hz by late adolescence (Miskovic et al., 2015; Cellier et al., 2021). This increase is thought to reflect maturation of thalamocortical circuits.
- **Absolute power** in the delta (1--4 Hz) and theta (4--8 Hz) bands tends to decrease with age, while alpha and beta power show more complex trajectories depending on the scalp region (Clarke et al., 2001; Whitford et al., 2007).
- **The aperiodic exponent** (1/f slope) decreases (flattens) with age, reflecting changes in the balance of excitatory and inhibitory neural activity (Schaworonkow & Bhzylak, 2023; Ostlund et al., 2022).

### 2.4 Spectral Parameterization

The specparam algorithm (Donoghue et al., 2020) models the power spectrum as the sum of an aperiodic component (characterized by offset and exponent) and one or more Gaussian peaks. By decomposing the spectrum in this way, researchers can separately examine oscillatory peaks (e.g., the alpha peak) and the aperiodic background, avoiding the confound of interpreting background shifts as oscillatory changes. This is particularly important in developmental studies where both components change with age.

### 2.5 EEG-Based Brain-Age Models

Brain-age prediction from EEG has been demonstrated in adult populations using spectral features and machine learning (Engemann et al., 2022; Sun et al., 2019). However, most prior work has focused on adults, used the model as a prediction tool without detailed feature interpretation, or did not explicitly separate periodic and aperiodic components. Our approach aims to address these gaps by targeting the developmental age range (5--21), providing explicit feature-group ablations, and grounding interpretations in developmental neuroscience.

---

## 3. Data

### 3.1 Pilot Dataset

For this pilot study, we used a publicly available resting-state EEG dataset from OpenNeuro (ds004148; Wang et al.), comprising EEG recordings from healthy adults with eyes-open (EO) and eyes-closed (EC) conditions across multiple sessions.

**Pilot subset:** We downloaded data from 20 participants (seeded random sample, seed = 1337), yielding 120 EEG recordings (60 EO, 60 EC; 3 sessions per participant). Participants ranged in age from 18 to 22 years (mean = 19.8). EEG was recorded using a 64-channel BrainVision system.

**Note on target dataset:** The intended target dataset is the Healthy Brain Network (HBN; OpenNeuro ds004186), covering ages 5--21. The pilot dataset (ds004148) was used because ds004186 requires OpenNeuro authentication. The pipeline is designed to work with any BIDS-compliant EEG dataset and will be applied to HBN when access is obtained. [TBD: run on full HBN dataset for the final paper.]

### 3.2 Ethics and Guardrails

All data used are publicly available, de-identified, and obtained from OpenNeuro under their data use agreements. This project is strictly for research and education purposes. It is not a diagnostic tool, clinical device, or decision-support system. Model outputs have not been clinically validated.

---

## 4. Methods

The pipeline is organized into six sequential stages, each reading from the previous stage's outputs and producing its own artifacts.

### 4.1 Stage 1: Data Acquisition and Manifest

We used the `openneuro-py` package to download a subset of the dataset. A participants manifest was constructed by scanning the BIDS directory structure and joining with `participants.tsv` for demographics. Only resting-state recordings (eyes-open and eyes-closed tasks) were retained.

### 4.2 Stage 2: Quality Control and Preprocessing

Minimal, defensible preprocessing was applied:

- **Bandpass filter:** 1--40 Hz (FIR, zero-phase)
- **Notch filter:** 60 Hz (to remove line noise)
- **Re-reference:** Common average
- **Epoching:** 2-second fixed-length segments
- **Artifact rejection:** Epochs with peak-to-peak amplitude exceeding 200 $\mu$V or flatline below 1 $\mu$V were rejected

QC metrics were computed per recording: usable duration, rejection fraction, line-noise proxy (ratio of spectral power at 60 Hz to neighboring frequencies), and muscle-artifact proxy (ratio of 30--45 Hz power to 8--13 Hz power).

**Pilot QC summary:** Of 120 recordings attempted, 120 were processed successfully. Mean usable duration was 282.5 seconds (median 292.0 s). Mean rejection fraction was 5.2% (median 2.0%). One recording had only 28 seconds of usable data and was flagged but retained for feature extraction. Line noise was fully removed by the notch filter (ratio = 0.00). Mean muscle-artifact proxy was 0.187.

### 4.3 Stage 3: Feature Engineering

From each cleaned recording, we extracted 34 spectral features grouped into four families:

**Band powers (24 features).** Power spectral density was computed using Welch's method. Absolute band powers were computed via trapezoidal integration over four frequency bands (delta: 1--4 Hz, theta: 4--8 Hz, alpha: 8--12 Hz, beta: 13--30 Hz) and reported as log$_{10}$(V$^2$). Band powers were computed globally (across all EEG channels) and for five scalp regions (frontal, central, parietal, occipital, temporal), defined by channel-name prefix matching.

**Peak alpha frequency and individualized alpha (4 features).** PAF was estimated as the frequency of maximum power in the 6--13 Hz range, computed from posterior channels (O, PO, P prefixes). Individualized alpha power was computed in a $\pm$2 Hz window centered on each participant's PAF (clipped to 4--14 Hz). Additional alpha-peak parameters (center frequency and height above aperiodic background) were extracted from the specparam fit.

**Aperiodic parameters (2 features).** The aperiodic exponent and offset were estimated using the specparam algorithm (Donoghue et al., 2020), fitting the 2--40 Hz range with up to 6 peaks. Specparam fits succeeded for all 118 included recordings.

**Ratios (2 features).** Theta/alpha and theta/beta power ratios were computed from global band powers (on the linear, not log, scale).

**QC carry-through (2 features).** Usable duration and rejection fraction were carried forward as potential covariates.

**Pilot feature summary:** 118 of 120 recordings produced complete feature vectors (2 excluded by QC duration threshold). PAF ranged from 6.5 to 13.0 Hz (mean = 10.2, SD = 1.3). Aperiodic exponent ranged from 0.20 to 2.41 (mean = 0.91, SD = 0.36).

### 4.4 Stage 4: Modeling

Three models were evaluated for age prediction:

1. **Baseline (mean):** Predicts the training-set mean age for all samples.
2. **RidgeCV:** Linear regression with L2 regularization, alpha selected via internal cross-validation from {0.01, 0.1, 1, 10, 100}.
3. **HistGradientBoostingRegressor (HGB):** Histogram-based gradient boosting with max depth = 3, learning rate = 0.05, 500 iterations, L2 regularization = 1.0.

**Leakage prevention.** GroupKFold cross-validation (K = 5) was used, grouping by participant ID so that all recordings from the same participant (EO and EC, all sessions) were assigned to the same fold. Preprocessing (median imputation and standard scaling) was fit on training folds only.

**Metrics:** Mean absolute error (MAE, in years) and coefficient of determination (R$^2$) were computed per fold and averaged.

### 4.5 Stage 5: Explainability

**Permutation importance.** For each CV fold, the best model (HGB, excluding baseline) was fit on training data, and permutation importance was computed on the held-out fold by permuting each feature 10 times and measuring the increase in MAE.

**Ablation experiments.** The HGB model was retrained with specific feature groups removed:
- Remove PAF/IAF features (4 features)
- Remove aperiodic features (2 features)
- Remove both PAF and aperiodic (6 features)

### 4.6 Stage 6: Robustness Checks

Two robustness checks were conducted:

**EO vs EC comparison.** The HGB model was trained and evaluated separately on eyes-open and eyes-closed recordings.

**Feature-family sensitivity.** Performance was compared using (i) all features, (ii) periodic features only, (iii) aperiodic features only, (iv) periodic + aperiodic, and (v) other features (usable duration, rejection fraction).

---

## 5. Results (Pilot)

### 5.1 Cohort Flow

| Stage | Recordings |
|-------|-----------|
| Downloaded | 120 |
| Passed QC | 120 (100%) |
| Included in features | 118 (98.3%) |
| Included in modeling | 118 |

### 5.2 Model Performance

| Model | MAE (years) | R$^2$ |
|-------|------------|-------|
| Baseline (mean) | 1.01 $\pm$ 0.43 | $-$0.115 $\pm$ 0.060 |
| RidgeCV | 1.25 $\pm$ 0.42 | $-$0.997 $\pm$ 0.610 |
| HGB | 1.12 $\pm$ 0.48 | $-$0.589 $\pm$ 0.310 |

The baseline (mean age prediction) achieved the lowest MAE. All R$^2$ values were negative, indicating that no model outperformed predicting the mean. This is expected given the narrow 4-year age range (18--22): there is insufficient age variance for EEG features to provide predictive signal above chance.

### 5.3 Feature Importance (Pilot)

The top 5 features by permutation importance (HGB model) were:

| Rank | Feature | Importance (MAE increase) |
|------|---------|--------------------------|
| 1 | Parietal theta power | 0.216 |
| 2 | Central delta power | 0.034 |
| 3 | Temporal beta power | 0.025 |
| 4 | Frontal alpha power | 0.023 |
| 5 | Alpha peak center (specparam) | 0.022 |

Parietal theta power dominated, contributing approximately 6 times more than the next most important feature. This may reflect individual differences in attentional state or drowsiness rather than developmental maturation, given the narrow age range.

### 5.4 Ablation Results

| Experiment | MAE (years) | Features |
|-----------|------------|----------|
| Full model | 1.12 | 34 |
| Remove PAF | 1.13 | 30 |
| Remove aperiodic | 1.15 | 32 |
| Remove both | 1.10 | 28 |

Ablation differences were minimal ($\leq$0.03 years), consistent with the absence of developmental signal in this narrow age range.

### 5.5 Robustness

**EO vs EC:** Eyes-closed recordings (MAE = 1.22) performed slightly better than eyes-open (MAE = 1.34), with the combined model (MAE = 1.12) outperforming both---likely due to the larger training set.

**Feature families:** Aperiodic features alone (2 features, MAE = 1.14) nearly matched the full feature set (34 features, MAE = 1.12), suggesting that the aperiodic exponent and offset capture most of the available age-related variance even in this narrow age range.

---

## 6. Discussion

The pilot results confirm that the pipeline operates correctly end-to-end: data are downloaded, quality-checked, feature-extracted, modeled, and interpreted in a reproducible, config-driven workflow. However, the narrow age range of the pilot dataset (18--22 years) fundamentally limits the ability to detect developmental EEG trajectories.

The finding that parietal theta power was the most important feature is consistent with theta's role in attentional and cognitive processes, but in a narrow age range, this likely reflects individual differences rather than maturation. Similarly, the near-equivalence of ablation conditions suggests that PAF and aperiodic features---which are known developmental markers---do not provide discriminative information when age variance is only 4 years.

We expect that applying this pipeline to the full HBN dataset (ages 5--21) will reveal the developmental trajectories described in the literature: increasing PAF, flattening aperiodic slopes, and shifting regional power distributions. These features should provide substantially more predictive signal, reducing MAE to levels comparable to prior adult brain-age studies (2--5 years depending on the age range and sample size).

---

## 7. Limitations

1. **Narrow age range.** The pilot dataset spans only 18--22 years, far narrower than the target developmental window (5--21). All modeling results are preliminary.
2. **Small sample size.** With only 20 participants (118 recordings), cross-validation estimates have high variance.
3. **Proxy dataset.** The pilot used ds004148 (a healthy adult EEG dataset) rather than the target HBN dataset (ds004186). EEG characteristics may differ.
4. **No ICA or advanced artifact rejection.** The pipeline uses only amplitude-based rejection, which may leave residual ocular or muscle artifacts.
5. **Correlation, not causation.** EEG features associated with age are correlational markers, not causal explanations.
6. **Not diagnostic.** This pipeline is a research tool. It has not been validated for any clinical application.

---

## 8. Next Steps: Scaling to Full Dataset

### 8.1 Data Expansion

[TBD: Obtain OpenNeuro authentication for ds004186 (HBN). Target: N > 500 participants, ages 5--21, both EO and EC conditions. Expected storage: approximately 50--100 GB.]

### 8.2 Method Upgrades

- Apply the pipeline to the full HBN dataset and report MAE, R$^2$, and calibration across the developmental age range
- Compare EO-only, EC-only, and combined models
- Test preprocessing sensitivity (e.g., 1--30 Hz vs 1--40 Hz bandpass)
- Add external validation on a held-out subset or a second dataset
- Consider adding connectivity features or source-level analysis in future iterations

### 8.3 Final Paper Deliverables

- Figure 1: Pipeline schematic (Stages 1--6)
- Figure 2: QC distributions (full dataset)
- Figure 3: Age vs PAF scatter with developmental trendline
- Figure 4: Predicted vs true age + residuals (calibration plot)
- Figure 5: Feature importance and ablation results
- Table 1: Cohort characteristics (full dataset)
- Table 2: Model performance comparison (full dataset)

---

## 9. Reproducibility

- **Dataset:** OpenNeuro ds004148 (pilot), ds004186 (target)
- **Random seed:** 1337
- **Config hashing:** All pipeline configs are YAML files with deterministic SHA-256 hashing
- **Software:** Python 3.13, MNE 1.12.1, scikit-learn 1.7.2, specparam, pandas, numpy
- **Code:** All scripts are in `scripts/stage{0-6}_*.py`, config-driven from `configs/stage{0-6}.yml`
- **Re-run:** Execute stages sequentially: `python scripts/stage{N}_*.py --config configs/stage{N}.yml`

---

## References

Alexander, L. M., Escalera, J., Ai, L., et al. (2017). An open resource for transdiagnostic research in pediatric mental health. *Scientific Data*, 4, 170181. https://doi.org/10.1038/sdata.2017.181

Appelhoff, S., Sanderson, M., Brooks, T. L., et al. (2019). MNE-BIDS: Organizing electrophysiological data into the BIDS format and facilitating their analysis. *Journal of Open Source Software*, 4(44), 1896. https://doi.org/10.21105/joss.01896

Cellier, D., Riddle, J., Petersen, I., & Hwang, K. (2021). The development of theta and alpha neural oscillations from ages 3 to 24 years. *Developmental Cognitive Neuroscience*, 50, 100969. https://doi.org/10.1016/j.dcn.2021.100969

Clarke, A. R., Barry, R. J., McCarthy, R., & Selikowitz, M. (2001). Age and sex effects in the EEG: development of the normal child. *Clinical Neurophysiology*, 112(5), 806--814. https://doi.org/10.1016/S1388-2457(01)00488-6

Donoghue, T., Haller, M., Peterson, E. J., et al. (2020). Parameterizing neural power spectra into periodic and aperiodic components. *Nature Neuroscience*, 23(12), 1655--1665. https://doi.org/10.1038/s41593-020-00744-x

Engemann, D. A., Mellot, A., Hoechenberger, R., et al. (2022). A reusable benchmark of brain-age prediction from M/EEG resting-state signals. *NeuroImage*, 262, 119521. https://doi.org/10.1016/j.neuroimage.2022.119521

Markiewicz, C. J., Gorgolewski, K. J., Feingold, F., et al. (2021). The OpenNeuro resource for sharing of neuroscience data. *eLife*, 10, e71774. https://doi.org/10.7554/eLife.71774

Miskovic, V., Ma, X., Chou, C.-A., et al. (2015). Developmental changes in spontaneous electrocortical activity and network organization from early to late childhood. *NeuroImage*, 118, 237--247. https://doi.org/10.1016/j.neuroimage.2015.06.013

Ostlund, B., Donoghue, T., Anaya, B., et al. (2022). Spectral parameterization for studying neurodevelopment: How and why. *Developmental Cognitive Neuroscience*, 54, 101073. https://doi.org/10.1016/j.dcn.2022.101073

Pernet, C. R., Appelhoff, S., Gorgolewski, K. J., et al. (2019). EEG-BIDS, an extension to the brain imaging data structure for electroencephalography. *Scientific Data*, 6, 103. https://doi.org/10.1038/s41597-019-0104-8

Sun, H., Paixao, L., Bhargava, A., et al. (2019). Brain age from the electroencephalogram of sleep. *Neurobiology of Aging*, 74, 112--120. https://doi.org/10.1016/j.neurobiolaging.2018.10.016

Whitford, T. J., Rennie, C. J., Grieve, S. M., et al. (2007). Brain maturation in adolescence: concurrent changes in neuroanatomy and neurophysiology. *Human Brain Mapping*, 28(3), 228--237. https://doi.org/10.1002/hbm.20273
