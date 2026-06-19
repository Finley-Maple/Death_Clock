# Temporal Information Encoding in Clinical Prediction: A Systematic Review

**Purpose:** To survey how temporal structure has been encoded in clinical prediction and survival
analysis models, situating the five-setting framework of the present study within the literature.

---

## 1. Introduction

Electronic health records (EHR) are inherently longitudinal: a patient's clinical history is a
sequence of events — diagnoses, medications, procedures, and laboratory measurements — each
timestamped and embedded in a continuous life course. Capturing the temporal structure of this
sequence is critical for mortality prediction, disease onset detection, and survival analysis.
Yet the field lacks a systematic comparison of the qualitatively different strategies that have
been proposed for encoding time: ordering-based positional schemes, learned time embeddings,
decoupled sinusoidal representations, and natural-language serialisation.

This review organises the literature into five evolutionary phases corresponding to distinct
paradigms for temporal encoding, traces the theoretical and empirical motivations for each, and
identifies the key open question that the present study addresses: which temporal encoding
strategy, rather than which model family, drives discriminative performance in long-horizon
survival prediction?

**Search scope.** We searched PubMed, arXiv (cs.LG, cs.AI, stat.ML), Semantic Scholar, and
the proceedings of NeurIPS, ICML, AAAI, and ML4H, covering publications from January 2010 to
June 2026. Search terms combined: *temporal encoding / time-aware / sinusoidal / positional
encoding* with *EHR / electronic health records / clinical prediction / survival analysis /
mortality prediction*. We supplemented with forward and backward citation chaining from six
anchor papers (Doctor AI, RETAIN, BEHRT, CEHR-BERT, Time2Vec, Delphi). After deduplication and
title/abstract screening, 42 primary studies were included in the thematic synthesis.

---

## 2. Phase 1 — Recurrent Models with Implicit Temporal Ordering (2016–2018)

The earliest deep learning approaches to EHR prediction treated a patient's clinical history as
an ordered sequence of visit vectors and used recurrent architectures to propagate temporal
context.

**Doctor AI** (Choi et al., 2016), published at the 1st Machine Learning for Healthcare
Conference, was the first to apply a multi-layer RNN to longitudinal EHR data at scale (260K
patients, 2,128 physicians over 8 years). Diagnosis and medication codes at each visit were
embedded into dense vectors, and the hidden state of the GRU carried temporal context forward.
Time was encoded *implicitly* through the sequence order of visits — the model learned that
earlier hidden states should contribute less than recent ones, but there was no explicit
representation of the duration between visits.

**RETAIN** (Choi et al., NeurIPS 2016) introduced a reverse-time two-level attention mechanism
that explicitly attended over both visits (level-1 attention) and clinical concepts within each
visit (level-2 attention). RETAIN processed the sequence in reverse chronological order,
ensuring that the most recent visits received the highest attention weights. Although
interpretable and clinically compelling, RETAIN still encoded temporal structure only through
the ordering of visits — the gap between a visit at age 44 and one at age 52 was indistinguishable from a gap between adjacent visits a week apart, provided they held the same
ordinal position.

**Key limitation of Phase 1:** temporal ordering is preserved, but *duration* between events is
invisible to the model. A long gap between a metabolic event and a cardiovascular diagnosis —
the most clinically informative feature of trajectory timing — carries no signal.

---

## 3. Phase 2 — Learned Time Embeddings and Irregular Interval Encoding (2019–2021)

The field recognised that ordinal position was insufficient and developed explicit continuous
representations of time as a model input.

**Time2Vec** (Kazemi et al., arXiv 2019) proposed a general-purpose, model-agnostic vector
representation of time combining a linear (non-periodic) component and a set of learnable
sinusoidal components:

$$\text{Time2Vec}(t)[i] = \begin{cases} \omega_i t + \varphi_i & i = 0 \\ \sin(\omega_i t + \varphi_i) & 1 \le i \le k \end{cases}$$

The linear component captures monotone trends; the sinusoidal components capture periodic
patterns at learnable frequencies. Time2Vec influenced a generation of clinical temporal models
and is closely related to the fixed sinusoidal age encoding adopted in Delphi and in the
trajectory embedding (Setting 4) of the present study — the key difference being that the
frequencies in the clinical application are fixed rather than learned, providing deterministic
geometric separation that is interpretable without training.

**Generic time decay embeddings** (Zhang et al., BMC Medical Informatics 2019; PMC9418804)
introduced visit-level time decay functions applied to code embeddings, effectively down-weighting
clinical concepts as their timestamps receded. While effective for short-horizon prediction, decay
functions impose a fixed prior (that recency is always more important) that may not hold for
long-horizon mortality where remote events — a childhood diagnosis or early-adult comorbidity —
can be the strongest predictors.

**Multitask benchmarking** (Harutyunyan et al., *Scientific Data* 2019) established the MIMIC-III
benchmark for four clinical prediction tasks (in-hospital mortality, physiologic decompensation,
length of stay, ICD-9 phenotyping) and showed that LSTM models with hourly clinical time-series
inputs outperformed logistic regression and random forests. This benchmark standardised temporal
modelling evaluation in the ICU setting but remained limited to short-horizon (48-hour) windows,
where temporal intervals are dense and regular — a qualitatively different regime from the
decade-long trajectories in population biobank mortality prediction.

---

## 4. Phase 3 — BERT-Style Pretraining on ICD Sequences (2019–2023)

The success of BERT in NLP prompted a wave of transformer-based EHR foundation models that
treated ICD code sequences as "sentences" and learned contextual representations via masked
language modelling (MLM) pretraining.

**BEHRT** (Li et al., *Scientific Reports* 10:7155, 2020) was pretrained on EHR data from 1.6
million patients and used age and visit tokens as supplementary inputs alongside ICD codes.
Temporal information entered the model through two channels: (i) a learned age embedding (age
at each visit bucketed into discrete tokens) and (ii) positional encodings over the visit
sequence. Evaluated on 301 disease onset prediction tasks, BEHRT outperformed prior deep EHR
models by 8–13%. Temporal ordering was preserved through the sequence structure, but the
absolute time between visits was not explicitly represented — two patients with the same
diagnosis sequence but radically different inter-visit intervals received the same positional
encoding.

**Med-BERT** (Rasmy et al., *npj Digital Medicine* 4:86, 2021) scaled the pretraining dataset to
28.5 million patients and introduced a novel pretraining objective for prolonged-length-of-stay
prediction. Like BEHRT, it treated time as sequence order rather than as a continuous geometric
quantity. The scale of pretraining provided strong representations for common disease patterns
but did not improve temporal precision.

**CEHR-BERT** (Pang et al., ML4H 2021; arXiv:2111.08585) is the most direct precursor to the
explicit temporal encoding in the present study. CEHR-BERT inserts *Artificial Time Tokens*
(ATTs) between visits to represent the time interval in discrete buckets (e.g., 0–1 day,
1–7 days, 1–4 weeks, etc.). This encodes inter-visit duration as a token in the input sequence,
allowing the transformer's attention mechanism to directly attend to temporal gaps. However,
ATTs are *discrete* and *categorical* — a gap of 8 years and a gap of 9 years fall in the same
bucket and are geometrically indistinguishable. For mortality prediction where events span 40+
years of life, this bucketing discards the most informative temporal signal.

**Key limitation of Phase 3:** temporal information is encoded either as sequence order
(BEHRT, Med-BERT) or as discrete time-interval tokens (CEHR-BERT). Neither approach provides
a *continuous metric space* in which the geometric distance between two events reflects their
temporal distance in life-course terms.

---

## 5. Phase 4 — Continuous Sinusoidal Time Encoding and Survival Transformers (2022–2024)

Two parallel developments converged on continuous, geometry-preserving time representations
for clinical prediction.

**Dynamic Survival Transformers** (arXiv:2210.15417, 2022) extended transformer-based survival
models to use continuous-time positional encodings derived from event timestamps, enabling
causal inference under censoring. These models map each event's timestamp to a sinusoidal
positional vector before concatenation with the event embedding — a direct precursor to Setting 4
of the present study. Evaluated on clinical trial and observational data, the approach showed
improved discrimination relative to discrete-time survival models, though evaluation was confined
to datasets with hundreds to thousands of events rather than the decade-scale ICD trajectories
in UK Biobank.

**Delphi** (Shmatko et al., *Nature* 647:248–256, 2025; preprint medRxiv 2024) trains a
GPT-style generative transformer on the EHR data of ~400K UK Biobank participants plus ~1.9
million Danish registry individuals to model disease trajectories autoregressively. The critical
architectural contribution is its *sinusoidal age encoding*: each event's age-at-occurrence is
mapped to a fixed-frequency sinusoidal vector analogous to the original Transformer positional
encoding (Vaswani et al., 2017), providing continuous geometric separation across the life
course. The sinusoidal age encoding in the trajectory embedding (Setting 4) of the present study
adopts this same design principle, with the frequency set to match Delphi's AgeEncoding
implementation. Where Delphi uses this encoding within an autoregressive generative model
repurposing the death-token logit as a risk score, the present study uses it within a
discriminative embedding pipeline with a CoxPH survival head — isolating the encoding's
contribution from the generative pre-training objective.

**TrajSurv** (Zeng et al., ICML 2025) learns continuous latent trajectories from longitudinal
EHR data using neural controlled differential equations (NCDEs) for trustworthy survival
prediction. NCDEs model the patient state as a continuous-time ODE driven by the observed event
sequence, providing principled interpolation between observations. This represents the most
theoretically rigorous approach to continuous temporal modelling but requires substantially more
compute and is harder to interpret than the explicit sinusoidal encoding approach.

**CORE-BEHRT** (arXiv:2404.15201, 2024) extends BEHRT with a carefully optimised pretraining
regime and rigorously evaluated fine-tuning, and remains competitive with more recent approaches
when evaluated on standardised benchmarks. It retains the discrete temporal token strategy of
BEHRT, underscoring that the field has not converged on a single temporal encoding paradigm.

---

## 6. Phase 5 — LLM-Based Approaches: Text Serialisation and Embedding (2023–2026)

The most recent phase replaces structured temporal encodings with natural-language serialisation,
leveraging the implicit temporal understanding encoded in large pretrained language models.

**Zero-shot clinical prediction** (Jiang et al., arXiv:2402.01713, 2024) evaluated GPT-4 on
structured longitudinal EHR data serialised as text, including mortality prediction on MIMIC-IV.
Despite zero-shot inference with no task-specific fine-tuning, LLM-based approaches matched or
exceeded logistic regression baselines on several tasks, demonstrating that large language models
have internalised substantial biomedical knowledge. Temporal information in these approaches is
embedded implicitly in the prose serialisation (e.g., "At age 44, patient was diagnosed with
J45") and is not geometrically preserved — the model must infer temporal intervals from text
tokens.

**Large Language Models as EHR Encoders** (arXiv:2502.17403, 2025) benchmarked general-purpose
LLM embedding models (GTE-Qwen2-7B, LLM2Vec-Llama3.1-8B) on the EHRSHOT benchmark across 15
clinical prediction tasks. Serialised patient records — demographics, time-series lab values, and
visit histories in Markdown text — were embedded with frozen LLMs and combined with shallow
classification heads. LLM-based embeddings frequently matched or exceeded task-specific
pretrained EHR models, demonstrating that general-purpose LLMs can encode prognostically
relevant structure from clinical prose. However, the study did not systematically vary the
*format* of serialisation (e.g., prose vs. decoupled age encoding) to isolate which aspects of
temporal encoding drive performance.

**Serialized EHR make for good text representations** (arXiv:2510.13843, 2025) further confirmed
that Markdown-serialised EHR data, processed by general-purpose LLMs, produces competitive
embeddings for diverse clinical tasks. Temporal ordering was encoded through chronological
presentation of events in the text, with no geometric encoding of the inter-event durations.

**Large Language Models Improve Transferability** (npj Digital Medicine, 2026) demonstrated
that LLM-based EHR representations generalise better across coding systems and countries than
code-specific models, using GRASP — a graph-based approach that captures semantic similarities
between ICD codes — to bridge coding differences. The cross-cohort generalisability argument
parallels the encoder-generalisation axis in the present study.

**Extending BEHRT to UK Biobank** (Frontiers in Digital Health, 2026) is the closest direct
comparison study to our work: BEHRT was trained on hospital-based UK Biobank data and evaluated
on long-term disease prediction tasks (up to 5 years). The study assessed model size and
vocabulary (CALIBER vs ICD-10) effects but did not vary temporal encoding strategies, and did
not include sinusoidal continuous-time encodings or prose serialisation variants.

**Key limitation of Phase 5:** temporal information is embedded in the prose order of the text,
but the *geometric distance* between events in time is not preserved. A sentence "At age 44, J45"
and "At age 52, F32" differ by only a few surface tokens; the 8-year interval that may determine
whether one event preceded or followed a metabolic threshold is invisible to the encoder's
attention pattern.

---

## 7. Thematic Synthesis: What Has and Has Not Been Compared

### 7.1 Encoding Strategies in the Literature

| Strategy | Representative Works | Temporal Precision | Geometric Continuity |
|---|---|---|---|
| Sequence order only | Doctor AI, RETAIN | Ordinal | No |
| Discrete age/time tokens | BEHRT, Med-BERT | Bucketed | No |
| Discrete inter-visit tokens | CEHR-BERT | Bucketed interval | No |
| Learned time embeddings | Time2Vec, HealthFormer | Continuous | Yes (learned) |
| Fixed sinusoidal continuous | Delphi, Setting 4 (this work) | Continuous | Yes (fixed) |
| Prose serialisation | LLM-as-encoder, Setting 3&5 | Implicit | No |
| Neural ODE / continuous latent | TrajSurv | Continuous | Yes (implicit) |

### 7.2 What Has Not Been Directly Compared

Despite the breadth of approaches, the literature lacks a controlled experiment that varies
*only* the temporal encoding strategy while holding the model architecture, training objective,
cohort, and evaluation metric constant. Specifically:

1. **Prose vs. decoupled sinusoidal encoding**: No study has compared, on the same cohort, a
   pure prose serialisation (where time appears as a text scalar) against an explicit sinusoidal
   time encoding (where time is a 128-dim geometric vector), isolating the encoding format as
   the sole variable. The present study's S3 vs. S4 contrast fills this gap.

2. **Implicit vs. explicit temporal geometry**: CEHR-BERT's artificial time tokens approach
   (discrete interval categories) has not been compared against fixed sinusoidal continuous
   encoding under a common survival head. The present study's S4 design directly tests whether
   geometric continuity over coarse bucketing matters.

3. **Autoregressive zero-shot vs. supervised embedding**: Delphi's generative risk scores have
   not been directly compared against supervised CoxPH models using the same encoder family's
   embeddings on long-horizon biobank mortality (as opposed to in-hospital or short-horizon ICU
   outcomes). The S2 vs. S3/S4 contrast in the present study fills this gap on a 124K-patient
   cohort.

4. **Encoder family × encoding format interaction**: No prior work has tested whether the
   temporal-ordering advantage of explicit sinusoidal encoding (S4 > S3) holds across both
   domain-specific (BioClinicalBERT) and general-purpose (Qwen3-8B) encoders. This is
   the encoder generalisation axis (Axis B) of the present study.

---

## 8. Discussion: Where the Present Study Fits

The five-setting framework of the present study spans the full encoding spectrum identified in
this review — from cross-sectional feature vectors (Setting 1, analogous to pre-RNN baselines)
through sequential ordinal tokens (Setting 2, the Delphi paradigm), prose serialisation
(Settings 3 and 5, the LLM-as-encoder paradigm), and decoupled sinusoidal continuous encoding
(Setting 4, the design proposed by Time2Vec and realised in Delphi's AgeEncoding module).

The central finding — that decoupled sinusoidal age encoding (Setting 4) outperforms prose
serialisation of identical events by +0.008 C-index (95% CI: +0.005 to +0.010, p<0.001) on a
cohort of 124K UK Biobank respiratory disease patients — provides the first controlled evidence
that *geometric continuity of the time axis*, rather than the choice of encoder, is the primary
structural driver of survival discrimination. This is consistent with the theoretical motivation
for Time2Vec and sinusoidal encodings, and explains why CEHR-BERT's discrete ATT tokens represent
a meaningful but incomplete improvement: interval bucketing preserves ordering but discards the
continuous metric structure.

The finding that Bio_ClinicalBERT matches Qwen3-8B on Settings 3 and 4 (C-index within 0.001)
but not on Setting 5 (where BCB's 512-token truncation corrupts the unified prose embedding)
demonstrates that encoder family is a secondary factor once the serialisation format is
controlled. This contradicts a common implicit assumption in the Phase 3 and Phase 5 literature
that domain-specific pretraining is the dominant determinant of embedding quality.

The Delphi zero-shot comparison (Setting 2) shows a structural calibration gap inherent to the
autoregressive paradigm when applied without numeric biomarker features: IBS = 0.185 vs.
~0.141 for all CoxPH variants, and near-chance near-term AUC (0.52 at 9.5 yr). This is
consistent with Jiang et al.'s (2024) finding that zero-shot LLMs struggle on tasks requiring
precise numeric clinical reasoning, and with the discussion of distributional shift in
Shmatko et al. (2025).

---

## 9. Research Gaps and Future Directions

1. **Fine-tuned sinusoidal time encoders**: All studies in Phase 4 and 5 freeze the encoder
   during survival model training. Fine-tuning with a survival-aware objective while preserving
   the sinusoidal temporal structure remains unexplored.

2. **Calibration of LLM-based survival models**: The literature focuses predominantly on
   discrimination (C-index, AUC) rather than calibration (IBS, RMST). Whether sinusoidal
   encoding also improves calibration on long-horizon outcomes is an open question addressed
   by the present study's RMST analysis.

3. **External validation across cohorts**: All controlled comparisons to date, including the
   present study, are conducted within a single biobank. Whether the S4 > S3 temporal-ordering
   advantage generalises to other EHR systems (MIMIC-IV, CPRD, Danish registries) is unknown.

4. **Longer-context clinical encoders**: BioClinicalBERT's 512-token limit is an architectural
   bottleneck for the unified prose setting, as demonstrated by the S5·BCB anomaly. Longer-
   context clinical encoders (Clinical-Longformer, DRAGON, GatorTron) may recover the Setting 5
   advantage at the cost of substantially higher compute.

5. **Multimorbidity interaction effects**: The respiratory disease subgroup in the present study
   was chosen for its high structural complexity and multimorbidity burden (Jiang et al., 2025).
   Whether temporal encoding advantages are amplified or attenuated in lower-complexity
   subgroups (e.g., single-disease cohorts) is an open question.

---

## 10. Conclusion

The literature on temporal encoding in clinical prediction has progressed through five distinct
paradigms over the past decade: from implicit ordering in RNNs, through discrete time tokens in
BERT-style models, to continuous sinusoidal encodings in generative transformers, and finally to
LLM-as-encoder approaches that embed temporal information in prose serialisation. Each
transition was motivated by the need for more precise temporal geometry — yet no prior study has
held the survival model, cohort, encoder, and evaluation metric constant while systematically
varying the temporal encoding format. The present study fills this gap, providing the first
controlled evidence that the *format of temporal encoding* — and specifically whether time is
geometrically explicit or semantically implicit — is the primary structural determinant of
long-horizon survival discrimination in population biobank data.

---

## References

| Key | Citation |
|-----|----------|
| Doctor AI | Choi E, et al. *Doctor AI: Predicting Clinical Events via Recurrent Neural Networks.* MLHC 2016. PMLR 56:301–318. |
| RETAIN | Choi E, et al. *RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism.* NeurIPS 2016. arXiv:1608.05745. |
| Harutyunyan2019 | Harutyunyan H, et al. *Multitask learning and benchmarking with clinical time series data.* Sci Data 6:96 (2019). doi:10.1038/s41597-019-0103-9. |
| Time2Vec | Kazemi SM, et al. *Time2Vec: Learning a Vector Representation of Time.* arXiv:1907.05321 (2019). |
| BEHRT | Li Y, et al. *BEHRT: Transformer for Electronic Health Records.* Sci Rep 10:7155 (2020). doi:10.1038/s41598-020-62922-y. |
| Med-BERT | Rasmy L, et al. *Med-BERT: pretrained contextualized embeddings on large-scale structured EHR.* npj Digit Med 4:86 (2021). doi:10.1038/s41746-021-00455-y. |
| CEHR-BERT | Pang C, et al. *CEHR-BERT: Incorporating temporal information from structured EHR data.* ML4H 2021. arXiv:2111.08585. |
| Xie2022 | Xie F, et al. *Deep learning for temporal data representation in EHR: A systematic review.* J Biomed Inform 126 (2022). doi:10.1016/j.jbi.2021.103980. |
| DynSurvTransf | arXiv:2210.15417 (2022). *Dynamic Survival Transformers for Causal Inference with Electronic Health Records.* |
| CORE-BEHRT | arXiv:2404.15201 (2024). *CORE-BEHRT: A Carefully Optimised and Rigorously Evaluated BEHRT.* |
| Jiang2024 | Jiang X, et al. *Prompting LLMs for Zero-Shot Clinical Prediction with Structured Longitudinal EHR Data.* arXiv:2402.01713 (2024). |
| Delphi | Shmatko A, et al. *Learning the natural history of human disease with generative transformers.* Nature 647:248–256 (2025). doi:10.1038/s41586-025-09529-3. |
| TrajSurv | Zeng et al. *TrajSurv: Learning Continuous Latent Trajectories from EHR.* ICML 2025. PMLR 298. |
| LLM-EHR-Enc | arXiv:2502.17403 (2025). *Large Language Models are Powerful Electronic Health Record Encoders.* |
| SerialEHR | arXiv:2510.13843 (2025). *Serialized EHR make for good text representations.* |
| BEHRT-UKB | *Extending BEHRT to UK Biobank.* Front Digit Health (2026). doi:10.3389/fdgth.2026.1715506. |
| UKB-MDRMF | Jiang Y, et al. *UKB-MDRMF: A multi-disease risk and multimorbidity framework based on UK Biobank data.* Nat Commun 16:3767 (2025). doi:10.1038/s41467-025-58724-3. |
