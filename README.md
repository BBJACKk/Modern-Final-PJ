# RADI605 Final Project Notebook

**Title:** Prediction of Cardiovascular Disease in Patients with Type 2 Diabetes Mellitus  
**Course task:** Longitudinal sequence modelling with PyTorch

## 1. Assignment interpretation and assumptions

Before I trained the models, I defined the main assumptions of the study so the whole workflow would stay clear and consistent.

1. I use `time_step` to represent the follow-up order for each patient.
2. I use a **5-step sequence** for every patient because the assignment requires five time steps.
3. If a patient has fewer than 5 records, I add padding. If a patient has more than 5 records, I keep the earliest 5 records.
4. I use `CVD` as the target variable and remove it from the input features to prevent target leakage.
5. I split the data at the **patient level**, not the row level, so one patient cannot appear in more than one dataset.
6. I handle missing values in two stages: first by within-patient forward/backward filling, and then by using statistics from the training set for any remaining missing values.
7. I use a simple RNN as the baseline model and a bidirectional GRU as the improved model.

.

## 0. Configuration

I place these settings at the beginning for two reasons. First, it improves reproducibility. Second, it helps me keep the same configuration for both the baseline model and the improved model. In particular, `MAX_STEPS = 5` directly follows the assignment requirement of using five time steps.

## 1. Load data

The raw dataset is loaded in its original long format, where each row represents one patient visit or one yearly record. Before starting the modelling process, I first check that the file is loaded correctly and that the date columns are read as dates instead of plain text.

The next cell loads the raw CSV file and reports the overall shape of the dataset together with the number of unique patients.

This first check helps me confirm that the dataset is longitudinal rather than purely cross-sectional. It also gives the main cohort size, which is useful later when I describe the sequence construction and the train, validation, and test split.

## 2. Problem definition and modelling assumptions

Before choosing features or training models, the prediction task must be defined clearly. In clinical machine learning, a weak problem definition can lead to leakage, unclear evaluation, or conclusions that are difficult to justify.

In this project, I define the task as **patient-level CVD detection from a fixed 5-step observation window**. For each patient, I build one sequence from the first five yearly records and assign one patient-level label based on the CVD information observed within that window.

## 3. Feature engineering

Feature engineering changes the raw clinical variables into inputs that can be used by the model while still keeping clinical meaning. The goal is not to use every column in the dataset, but to keep variables that are reasonable for cardiovascular risk prediction in patients with type 2 diabetes.

My main feature-engineering choices are as follows:

- I convert `dob` and `visit_date` into **age**, because age is easy to interpret clinically while raw date values are not.
- I encode **sex** numerically so it can be used by the model.
- I remove patient identifiers and clear leakage variables from the predictor list.
- I keep predictors that are clinically relevant to cardiovascular risk in diabetes, such as demographics, body size, blood pressure, diabetic complications, medication use, lipid markers, glucose control, and renal function.

## 4. Patient-level split

The dataset is divided into training, validation, and test sets at the patient level. This is a very important step for longitudinal healthcare modelling.

I split the data at the **patient level** instead of the row level.
If rows were split directly, visits from the same patient could appear in both training and testing. That would leak patient-specific information across datasets and make the model performance look better than it really is. To avoid this problem, I first identify unique patients, create a patient-level label, and then split using patient IDs only.

The final ratio is **70% training, 15% validation, and 15% test**.I also use stratification on the patient-level label so the class distribution stays reasonably similar across the three subsets.

## 5. Fit preprocessing statistics on the training set only

All preprocessing statistics should be learned from the training set only. This includes imputation values and standardization parameters.

The preprocessing strategy is simple, practical, and easy to explain.

1. **Within-patient forward fill / backward fill for numeric variables**  
   For repeated measurements such as blood pressure, creatinine, HbA1c, and lipid values, the nearest values from the same patient are usually more reasonable than using information from the whole cohort immediately.

2. **Binary indicator imputation with zero**  
   For medication and diagnosis flags stored as 0/1 variables, I treat missing values as absence unless the dataset gives a different rule.

3. **Fallback to training-set medians**  
   If a numeric feature is still missing after within-patient filling, I replace it with the median from the training set. I use the median because many laboratory values are skewed and may contain outliers.

4. **Standardization from training-set statistics**  
   I standardize continuous features using the training-set mean and standard deviation, and then apply the same transformation to the validation and test sets.

All preprocessing statistics are fitted on the training set only. This keeps the evaluation fair and reduces leakage from the validation or test data.

## 6. Build 5-step sequences

After the preprocessing statistics are prepared, the row-level data are changed into fixed-length patient sequences.
After preprocessing, I convert the visit-level table into fixed-length patient sequences.

For each patient, I sort the records by `time_step`, keep the first five observations, and pad shorter histories when necessary. The sequence-building function returns three outputs:

- **`X`**: the model input with shape `[N, 5, F]`
- **`y`**: the patient-level binary target
- **`seq_len`**: the true observed length before padding

I keep the true sequence length because the improved model uses packed sequences. This helps the recurrent layer focus on real observations instead of padded rows.

## 7. Dataset and DataLoader

After sequence construction, the arrays are wrapped into a PyTorch dataset and data loaders. This gives the format needed for mini-batch training.

The custom dataset stores three items for each patient: the fixed-length sequence, the binary label, and the observed sequence length.

The DataLoader then groups these samples into mini-batches during training. This makes memory use more manageable and allows the same training loop to be used for both the baseline model and the improved packed-sequence model.

## 8. Model definitions

Two recurrent models are implemented: a baseline RNN and an improved bidirectional GRU. The comparison is designed to show not only whether performance changes, but also why the choice of architecture matters for longitudinal clinical data.

### Baseline model: vanilla RNN

I use a simple vanilla RNN as the baseline because it provides a clear reference point for the project. The baseline is not expected to be the strongest model. Its purpose is to show the performance of a basic recurrent architecture on this task.

### Improved model: bidirectional GRU with dropout and packed sequences

For the improved model, I replace the plain recurrent unit with a bidirectional GRU and keep dropout in the network. I also use packed sequences so padded time steps have less effect on learning.

This change is reasonable for this task. A GRU has gating mechanisms that usually handle noisy longitudinal signals better than a plain RNN. In addition, the bidirectional structure allows the model to summarize the full five-step observation window before making the final prediction.

## 9. Evaluation metrics

A single accuracy value is not enough for evaluating a clinical prediction model. For this reason, I report a group of complementary metrics.

The evaluation function reports several metrics: AUROC, AUPRC, accuracy, precision, recall, F1-score, specificity, and the confusion-matrix counts.

I treat **AUROC** and **AUPRC** as the most important summary measures, especially AUPRC, because the positive class is relatively uncommon and accuracy alone may be misleading. I also keep recall and specificity because this is a clinical risk prediction task: missing higher-risk patients and over-flagging lower-risk patients are both important errors, so one metric alone is not enough.

## 10. Training utilities

This section defines the training loop, validation loop, probability prediction function, and the overall fitting routine used by both models.

The training utility is shared by both models so that the final comparison is as fair as possible.

Several implementation choices are important here:

- I use **`BCEWithLogitsLoss`** for numerically stable binary classification.
- I apply **positive-class weighting** to reduce the effect of class imbalance.
- I use **Adam** for optimization.
- I save the best validation result together with the corresponding model weights.
- I record the training history so I can inspect whether learning is stable or clearly overfitted.

Because both models use the same training framework, any major difference in performance should mainly come from the model architecture instead of inconsistent training settings.

## 11. Baseline model training

The baseline model is trained first so that it can serve as the reference model for the rest of the project.

The baseline-training cell saves the model weights, draws the learning curves, and reports the final test metrics.

These results are important because they provide the reference point for the project. If the improved model cannot perform better than this baseline under the same data split and preprocessing pipeline, then the architectural improvement would not be convincing.

## 12. Hyperparameter tuning for the improved model

A small hyperparameter search is carried out for the improved model using the validation set. The purpose is not to test every possible setting, but to show a structured model-selection process.

For the improved model, I tune the main hyperparameters that are most likely to affect performance: hidden size, number of recurrent layers, dropout, learning rate, batch size, and weight decay.

I rank candidate models mainly by **validation AUPRC** and then by **validation AUROC**. I choose this ranking because the positive class is clinically important and not evenly balanced with the negative class. After tuning, I retrain the final BiGRU using the selected configuration and evaluate it on the untouched test set.

## 13. Train the final improved model

After selecting the best hyperparameters, the improved model is retrained and evaluated on the held-out test set. This keeps the test set untouched during model selection.

The final improved model uses the best BiGRU configuration found during tuning and is evaluated with exactly the same pipeline used for the baseline.

This direct comparison is important. The train, validation, and test split, preprocessing steps, loss design, and evaluation metrics are all kept the same, so the final difference is mainly about the effect of the improved recurrent architecture.

## 14. Final comparison

The purpose of this section is to place both models in one summary table and to show the final classification behaviour of the improved model.

The final comparison table answers the main question of the project: whether the improved recurrent model gives a meaningful advantage over the baseline RNN for patient-level CVD detection.

When I interpret the table, I focus on the metrics that matter most for this task:

- whether AUROC improves,
- whether AUPRC improves,
- whether recall improves enough to justify any loss in precision or specificity,
- and whether the confusion matrix shows a more acceptable clinical error pattern.

For this type of problem, a model is not automatically better just because one number increases. The trade-off between finding more positive cases and creating more false positives also needs to be discussed.

## 15. Discussion

This final section explains the results instead of only repeating the numbers.

The improved BiGRU performed better than the baseline RNN on the main discrimination metrics. On the test set, AUROC increased from **0.7656** to **0.7965**, and AUPRC increased from **0.2979** to **0.3381**. Recall also improved from **0.6400** to **0.7500**, which means the improved model detected a larger proportion of patients in the positive class.

I think this improvement is meaningful for two reasons. First, the GRU architecture is more suitable for longitudinal clinical data than a plain RNN because its gating mechanism helps the model decide what information should be kept or updated over time. Second, the bidirectional structure can summarize the full five-step window more effectively, while packed sequences reduce the effect of padded time steps.

However, the improved model was not better on every metric. The baseline RNN achieved higher accuracy, precision, and specificity, so it behaved in a more conservative way. In other words, the BiGRU found more positive cases, but it also generated more false positives. For this reason, I would not say that the BiGRU is better in every situation. A more balanced interpretation is that the BiGRU gives a better **screening-oriented** performance, while the baseline gives a more conservative error pattern.

This trade-off is important in healthcare. If the goal is early risk identification, higher recall and better AUPRC may be more valuable than a small gain in overall accuracy. On the other hand, if false alarms create extra workload for clinicians, then the lower specificity of the BiGRU also becomes important.

There are still some limitations. This study utilized only a single dataset and did not include external validation. The interpolation rules are based on theoretical assumptions, and the fixed five-step window may exclude useful information from longer medical histories. Furthermore, the model relies solely on structured follow-up data and does not incorporate clinical records, imaging studies, or other contextual information. Therefore, this should be considered merely a proof of concept, and further in-depth validation and investigation are required before it can be applied in clinical practice.

### Conclusion

Overall, this project shows that it is useful to model the data as patient-level longitudinal sequences, and that the proposed BiGRU provides a practical improvement over the baseline RNN under the same preprocessing and evaluation pipeline. The main gains appear in AUROC, AUPRC, and recall, which makes the improved model more suitable when the goal is to identify more patients at cardiovascular risk within the observed follow-up window.

At the same time, the results also show a cost: the improved model has lower specificity and precision than the baseline. This means the final model choice should depend on the clinical objective. For this project, I consider the BiGRU the stronger overall model because it provides better discrimination and case-finding ability for an imbalanced clinical prediction task.
