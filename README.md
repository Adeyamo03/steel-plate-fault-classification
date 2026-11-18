# Steel Plate Fault Classification — My Manufacturing Machine Learning Experiment

## Project overview
I developed this machine-learning experiment to investigate whether numerical measurements of existing steel plate faults can be used to predict their fault categories. I trained and compared a **Decision Tree** and a **Random Forest** classifier using a real manufacturing dataset. My goal was to practice a reproducible data science workflow and explore how machine learning can support manufacturing quality analysis.

## Research question
**Can I use numerical measurements of an already identified steel plate fault to predict which of seven fault categories it belongs to?**

## Dataset
I used the [Steel Plates Faults dataset from the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/198/steel+plates+faults) (dataset ID 198). It includes **1,941 records, 27 input features, and seven fault categories**: Bumps, Dirtiness, K_Scratch, Other_Faults, Pastry, Stains, and Z_Scratch.

Because the dataset contains records of **existing faults**, my models classify the **type of fault**. They do **not** distinguish healthy steel plates from defective ones.

## Tools and technologies
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Google Colab
- `ucimlrepo` for retrieving the dataset

## My methodology
1. **Loaded the dataset.** I retrieved the UCI data and examined its features and fault labels.
2. **Explored the data.** I inspected the class distribution, data types, and missing values to understand the dataset before modeling.
3. **Split the data.** I used an **80/20 stratified train-test split** to preserve approximately the same proportion of fault categories in the training and test sets.
4. **Trained two classifiers.** I fit a Decision Tree and a Random Forest to the training data.
5. **Evaluated the models.** I compared held-out test accuracy and macro-F1, and examined classification reports and confusion matrices to identify where the models made mistakes.
6. **Visualized the results.** I created a model-comparison chart and inspected the classification patterns. I also used the notebook to explore model feature importance and batch predictions.

## My results
On my **389-record held-out test set**, **Random Forest** outperformed **Decision Tree** on both accuracy and macro-F1.

| Model | Test accuracy | Test macro-F1 | Correct predictions |
|---|---:|---:|---:|
| Decision Tree | 72.24% | 0.7664 | 281 / 389 |
| **Random Forest** | **79.69%** | **0.8034** | **310 / 389** |

I selected **macro-F1** as an important evaluation measure because it gives each fault category equal weight rather than allowing the most common classes to dominate the score.

### Error analysis
I reviewed the Random Forest confusion matrix to identify the categories that were most difficult to classify. **Pastry had the lowest recall**: the model correctly classified **17 of 32** Pastry cases, a recall of **53.13%**. It incorrectly predicted **12** Pastry cases as **Other_Faults** and **3** as **Bumps**.

This showed me why overall accuracy is not sufficient on its own. Even though Random Forest achieved nearly 80% accuracy, its performance was noticeably weaker for some fault types.

## Limitations
- I used a dataset of **known faults**, so I cannot claim that my model detects whether a steel plate is defective in the first place.
- My reported results come from **one train-test split**; they do not establish how stable performance would be across different splits.
- The dataset may not represent other factories, materials, equipment, or production conditions. I have **not** validated the models on a different manufacturing facility's data.
- A model's feature importance does not, by itself, establish that a feature causes a specific fault.
- This is a research-learning and portfolio experiment, **not** a deployed industrial inspection system.

## What I would test next
I would use **stratified cross-validation** to assess how consistently the models perform, then apply **hyperparameter tuning on training data only** to explore improvements. I would pay particular attention to improving recall for the Pastry category, and I would test on an independent industrial dataset if one became available. These steps could improve confidence in model reliability, although improvement is not guaranteed.

## How to reproduce my experiment
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `Samuel_Esan_Steel_Plate_Experiment_First_Person.ipynb` using **File → Upload notebook**.
3. Select **Runtime → Run all**. The notebook installs `ucimlrepo` to retrieve the source data.
4. Review the model metrics, plots, confusion matrices, and conclusions. If results differ across executions, report the results from the actual run rather than reusing the numbers above.

## What I learned
Through this experiment, I practiced supervised classification with a real industrial dataset, compared a single Decision Tree against an ensemble method, and learned to interpret class-specific errors rather than relying on overall accuracy alone. I also gained experience communicating experimental limitations and identifying next steps for stronger validation.

## Dataset attribution
UCI Machine Learning Repository: [Steel Plates Faults (dataset 198)](https://archive.ics.uci.edu/dataset/198/steel+plates+faults).
