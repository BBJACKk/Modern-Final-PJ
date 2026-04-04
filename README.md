# RADI605 Final Project Notebook

**Title:** Prediction of Cardiovascular Disease in Patients with Type 2 Diabetes Mellitus  
**Course task:** Longitudinal sequence modelling with PyTorch


## 1. Assignment interpretation and assumptions

Before training any model, I fixed the problem setting so the rest of the notebook would be consistent and reproducible.

1. I treat `time_step` as the longitudinal order of follow-up for each patient.
2. I use a **5-step sequence** for every patient because the assignment explicitly asks for five time steps.
3. If a patient has fewer than 5 observations, I pad the remaining steps; if a patient has more than 5 observations, I keep the earliest 5 records.
4. I use `CVD` as the prediction target and remove it from the input features to avoid target leakage.
5. I split the data at the **patient level**, not the row level, so the same patient cannot appear in more than one subset.
6. I handle missing data in two stages: within-patient forward/backward filling first, then training-set statistics for any values that are still missing.
7. I use a simple RNN as the baseline and a bidirectional GRU as the improved model.

These assumptions are stated clearly here because they affect every later step, especially preprocessing, sequence construction, and model evaluation.

## 0. Configuration

This section defines the basic experimental environment. The purpose is to make the analysis reproducible and easy to rerun on another machine or in a private GitHub repository.

The next cell defines the file paths, random seeds, hardware setting, and the fixed sequence length used throughout the project.

I kept these settings at the top for two reasons. First, they make the notebook easier to reproduce. Second, they make it easier to check that the same configuration was used for both the baseline and the improved model. In particular, `MAX_STEPS = 5` is important because it directly enforces the project requirement of using a fixed five-step longitudinal input.

## 1. Load data

The raw dataset is first loaded in its original long format, where each row corresponds to one patient visit or yearly step. Before any modelling starts, it is important to verify that the file is read correctly and that date fields are parsed as dates rather than plain text.

The next cell loads the raw CSV file and reports the overall shape of the dataset together with the number of unique patients.

I use this first inspection to confirm that the data are longitudinal rather than purely cross-sectional. It also gives the basic cohort size that I refer to later when I describe sequence construction and the train/validation/test split.

## 2. Problem definition and modelling assumptions

Before selecting features or building models, the prediction target must be defined carefully. In clinical machine learning, weak problem definition often leads to leakage, ambiguous evaluation, or claims that are difficult to defend.

In this project, I formulate the task as **patient-level CVD detection from a fixed 5-step observation window**. For each patient, I build one sequence from the first five yearly records and assign one patient-level label based on the CVD information observed within that window.

I chose this formulation for three main reasons:

1. It matches the assignment requirement of using five time steps.
2. It preserves the temporal order of repeated follow-up measurements.
3. It avoids using information beyond the defined observation window.

This is therefore a sequence classification problem, not a row-by-row classification problem.

## 3. Feature engineering

Feature engineering converts raw clinical variables into model-ready inputs while keeping the design clinically meaningful. The goal here is not to use every available column blindly, but to select variables that plausibly relate to cardiovascular risk in patients with type 2 diabetes mellitus.

The next cell creates derived variables and defines the final feature set used for modelling.

My main feature-engineering decisions are straightforward:

- I convert `dob` and `visit_date` into **age**, because age is clinically interpretable while raw date strings are not.
- I encode **sex** numerically so it can be used by the model.
- I remove patient identifiers and obvious leakage variables from the predictor list.
- I keep predictors that are clinically reasonable for cardiovascular risk in diabetes, including demographics, body size, blood pressure, diabetic complications, medication exposure, lipid markers, glucose control, and renal function.

I did not try to keep every available column. Instead, I kept variables that I could justify clinically and that were usable within a reproducible preprocessing pipeline.

## 4. Patient-level split

The dataset is split into training, validation, and test sets at the patient level. This is a critical methodological step in longitudinal healthcare modelling.

I split the dataset at the **patient level** rather than the row level. This is one of the most important design choices in the notebook.

If I had split rows directly, visits from the same patient could have appeared in both training and testing. That would leak patient-specific information across subsets and make the performance look better than it really is. To avoid that problem, I first identify unique patients, create a patient-level label, and then perform the split on patient IDs only.

The final proportions are **70% training, 15% validation, and 15% test**, which follow the assignment instructions. I also use stratification on the patient-level label so the class balance stays reasonably similar across the three subsets.

## 5. Fit preprocessing statistics on the training set only

All preprocessing statistics must be learned from the training set only. This includes imputation values and standardization parameters.

The preprocessing strategy is intentionally conservative and easy to explain.

1. **Within-patient forward fill / backward fill for numeric variables**  
   For repeated measurements such as blood pressure, creatinine, HbA1c, and lipid values, the closest observations from the same patient are usually more sensible than borrowing information from the whole cohort immediately.

2. **Binary indicator imputation with zero**  
   For medication and diagnosis flags stored as 0/1 indicators, I treat missing values as absence unless the dataset explicitly says otherwise.

3. **Fallback to training-set medians**  
   If a numeric feature is still missing after within-patient filling, I replace it with the median from the training set. I use the median because many laboratory variables are skewed and may contain outliers.

4. **Standardization from training-set statistics**  
   I standardize the continuous features using the training-set mean and standard deviation, then apply the same transformation to validation and test data.

All preprocessing statistics are fitted on the training set only. This is necessary to keep the evaluation fair and to avoid leakage from validation or test data.

## 6. Build 5-step sequences

Once preprocessing statistics are available, the row-wise data are transformed into fixed-length patient sequences. This is the core step that converts the raw dataset into a sequence-learning problem.

After preprocessing, I convert the visit-level table into fixed-length patient sequences.

For each patient, I sort records by `time_step`, keep the first five observations, and pad shorter histories when needed. The sequence-building function returns three objects:

- **`X`**: the model input with shape `[N, 5, F]`
- **`y`**: the patient-level binary target
- **`seq_len`**: the true observed length before padding

I keep the true sequence length because the improved model uses packed sequences. That way, the recurrent layer pays attention to real observations rather than learning from padded rows.

## 7. Dataset and DataLoader

After sequence construction, the arrays are wrapped in a PyTorch dataset and data loaders. This creates the interface required for minibatch training.

The custom dataset stores three pieces of information for each patient: the fixed-length sequence, the binary label, and the observed sequence length.

The DataLoader then batches these samples for mini-batch training. This keeps memory use manageable and allows the same training loop to work for both the baseline model and the improved packed-sequence model.

## 8. Model definitions

Two recurrent models are implemented in this notebook: a baseline RNN and an improved bidirectional GRU. The comparison is designed to show not only whether performance changes, but also why architecture choices matter in longitudinal health data.

### Baseline model: vanilla RNN

I use a simple vanilla RNN as the baseline because it gives a clear reference point for the rest of the project. The baseline is not meant to be the strongest model; its role is to show what happens when the sequence is modelled with the most basic recurrent architecture.

### Improved model: bidirectional GRU with dropout and packed sequences

For the improved model, I replace the vanilla recurrent unit with a bidirectional GRU and keep dropout in the architecture. I also use packed sequences so padded timesteps have less influence on learning.

This change is reasonable for this task. A GRU has gating mechanisms that usually handle noisy longitudinal signals better than a plain RNN, and the bidirectional structure lets the model summarize the full five-step observation window before making the final classification.

## 9. Evaluation metrics

A single accuracy value is not sufficient for evaluating a clinical risk model. For this reason, the notebook reports a group of complementary metrics.

The evaluation function reports several metrics: AUROC, AUPRC, accuracy, precision, recall, F1-score, specificity, and the confusion-matrix counts.

I treat **AUROC** and **AUPRC** as the most important summary metrics, especially AUPRC, because the positive class is relatively uncommon and accuracy alone would be misleading. I also keep recall and specificity because this is a clinical prediction task: missing higher-risk patients and over-flagging lower-risk patients are both important errors, so a single metric is not enough.

## 10. Training utilities

This section defines the training loop, validation loop, probability prediction function, and the overall fitting routine used by both models.

The training utility is shared by both models so that the final comparison is as fair as possible.

A few implementation choices matter here:

- I use **`BCEWithLogitsLoss`** for numerically stable binary classification.
- I apply **positive-class weighting** to reduce the effect of class imbalance.
- I use **Adam** for optimization.
- I keep track of the best validation result and save the corresponding model weights.
- I store the training history so I can inspect whether learning is stable or obviously overfitted.

Using the same training framework for both models means that the main difference in performance should come from the model architecture rather than from inconsistent training settings.

## 11. Baseline model training

The baseline model is trained first so that it can serve as the reference model for the rest of the project.

The baseline-training cell saves the model weights, produces the learning curves, and reports the final test metrics.

These results matter because they define the reference point for the project. If the improved model cannot do better than this baseline on the same split and under the same preprocessing pipeline, then the architectural change would not be convincing.

## 12. Hyperparameter tuning for the improved model

A small hyperparameter search is carried out for the improved model using the validation set. The aim is not to exhaust every possible configuration, but to demonstrate a structured model-selection process.

For the improved model, I tune the main hyperparameters that are most likely to affect performance: hidden size, number of recurrent layers, dropout, learning rate, batch size, and weight decay.

I rank candidate models primarily by **validation AUPRC** and then by **validation AUROC**. I chose this ranking because the positive class is clinically important and not evenly balanced with the negative class. After tuning, I retrain the final BiGRU using the selected configuration and evaluate it on the untouched test set.

## 13. Train the final improved model

Once the best hyperparameters are selected, the improved model is retrained and evaluated on the held-out test set. This ensures that the test set remains untouched during model selection.

The final improved model uses the best BiGRU configuration found during tuning and is evaluated with exactly the same pipeline used for the baseline.

This one-to-one comparison matters. The train/validation/test split, preprocessing steps, loss design, and reporting metrics are all held constant, so the comparison is really about the effect of the improved recurrent architecture.

## 14. Final comparison

The purpose of this section is to bring both models together in a single summary table and visualize the final classification behaviour of the improved model.

The final comparison table answers the main question of the project: whether the improved recurrent model gives a meaningful advantage over the baseline RNN on patient-level CVD detection.

When I interpret the table, I focus on the metrics that matter most for this problem:

- whether AUROC improves,
- whether AUPRC improves,
- whether recall improves enough to justify any loss in precision or specificity,
- and whether the confusion matrix shows a more acceptable clinical error pattern.

For this kind of task, a model is not automatically better just because one number goes up. The trade-off between finding more positive cases and generating more false positives also has to be discussed.

## 15. Discussion

This final section interprets the numerical results rather than just repeating them.

The improved BiGRU performed better than the baseline RNN on the main discrimination metrics. On the test set, AUROC increased from **0.7656** to **0.7965**, and AUPRC increased from **0.2979** to **0.3381**. Recall also improved from **0.6400** to **0.7500**, which means the improved model identified a larger share of patients in the positive class.

I consider this improvement meaningful for two reasons. First, the GRU architecture is better suited to longitudinal clinical data than a plain RNN because its gating mechanism helps control what information should be retained or updated over time. Second, the bidirectional structure summarizes the full observed five-step window more effectively, while packed sequences reduce the influence of padded timesteps.

At the same time, the improved model did not dominate the baseline on every metric. The baseline RNN had higher accuracy, precision, and specificity, so it behaved more conservatively. In other words, the BiGRU found more positive cases, but it also produced more false positives. I therefore would not claim that the BiGRU is universally superior. A more accurate interpretation is that the BiGRU achieved a better **screening-oriented** profile, whereas the baseline preserved a more conservative error pattern.

This trade-off is important in a healthcare setting. If the purpose of the model is early risk identification, higher recall and better AUPRC may be more valuable than a small gain in overall accuracy. However, if false alarms create a large downstream burden, then the drop in specificity would also matter.

There are still several limitations. This is a single-dataset study with no external validation, the imputation rules are assumption-based, and the fixed five-step window may discard useful information from longer histories. In addition, the model uses structured tabular follow-up data only and does not incorporate unstructured notes, imaging, or other sources of clinical context. For these reasons, I interpret the project as a proof-of-concept for longitudinal CVD risk modelling rather than a ready-to-deploy clinical decision support tool.

### Conclusion

Overall, the project shows that modelling the data as patient-level longitudinal sequences is useful, and that the proposed BiGRU offers a practical improvement over the baseline RNN under the same preprocessing and evaluation pipeline. The gain is strongest on AUROC, AUPRC, and recall, which makes the improved model more suitable when the main goal is to identify more patients at cardiovascular risk within the observed follow-up window.

At the same time, the results also show a cost: the improved model is less specific and less precise than the baseline. This means the model choice should depend on the clinical objective. For this assignment, I regard the BiGRU as the stronger overall model because it provides better discrimination and case-finding ability on an imbalanced clinical prediction task.
