# Customer Behaviour Prediction — Delivery / Logistics Domain

**Apziva — Customer Happiness Prediction**

> Predicting customer satisfaction for a delivery & logistics startup (ACME) using an ordinal survey dataset. This repository contains exploratory data analysis, feature engineering, model training and evaluation, and artifacts (reports and figures) that demonstrate end-to-end work from raw data to actionable insights.

---

## 📌 Project Summary

This project analyzes the `ACME-HappinessSurvey2020.csv` dataset to build a classifier that predicts whether a customer is **happy (Y=1)** or **unhappy (Y=0)** following a delivery. All input features (X1–X6) are ordinal survey items (1–5) capturing delivery punctuality, item correctness, completeness of order, price perception, courier satisfaction and app usability.

The goals were:
- Thorough Exploratory Data Analysis (EDA) and data-quality checks.
- Feature engineering and outlier/skewness analysis.
- Train and compare multiple classifiers (Random Forest, Logistic Regression, k-NN).
- Identify the most important drivers of customer happiness.
- Produce clear, reproducible artifacts (CSV reports and plots) for stakeholders.

---

## 📁 Repository Structure

```
happycustomer/
├── data/
│   └── ACME-HappinessSurvey2020.csv    # Original dataset (not tracked due to size/privacy)
├── reports/
│   ├── feature_stats.csv
│   ├── feature_stats_with_outliers.csv
│   ├── feature_stats_with_outliers_and_skewness.csv
│   └── figures/
│       ├── correlation_matrix.png
│       ├── customer_types.png
│       ├── feature_importance.png
│       └── X1_distribution.png ... X6_distribution.png
├── setup/
│   └── requirements.txt
├── environment.yml
├── src/
│   └── HappyCustomer.ipynb            # Main analysis notebook (EDA + modeling)
└── Readme.md                          # Original short README
```

---

## 🔎 Data Description

The dataset contains 126 observations and 7 columns:
- **Y** — target (0 = unhappy, 1 = happy)
- **X1** — My order was delivered on time (1–5)
- **X2** — Contents of my order was as I expected (1–5)
- **X3** — I ordered everything I wanted to order (1–5)
- **X4** — I paid a good price for my order (1–5)
- **X5** — I am satisfied with my courier (1–5)
- **X6** — The app makes ordering easy for me (1–5)

> Note: All features are ordinal with no missing values in the provided dataset.

---

## 🧭 Project Flow — Step by Step

1. **Data loading & sanity checks**
   - Load CSV into a pandas DataFrame, inspect schema and basic statistics.
   - Verify there are no missing values and confirm value ranges for ordinal items.

2. **Exploratory Data Analysis (EDA)**
   - Class balance check (happy vs unhappy).
   - Distribution plots (KDE / histograms) for each feature split by target.
   - Correlation matrix and heatmap to identify strongly related features.
   - Outlier detection (IQR method) and skewness measurements.
   - Save stats and figures to `reports/`.

3. **Feature analysis & selection**
   - Evaluate relationships between X1–X6 and the target.
   - Identify redundant or low-importance features (X6 identified as least important).

4. **Modeling**
   - Train and compare several classifiers:
     - RandomForestClassifier
     - LogisticRegression (with PowerTransformer + StandardScaler)
     - KNeighborsClassifier
   - Use an 80/20 train-test split and evaluate via accuracy and F1-score.
   - Use feature importance (Random Forest) and logistic coefficients for interpretability.

5. **Evaluation & reporting**
   - Report results (accuracy/F1) and visualize feature importance.
   - Save figures and CSV summary reports under `reports/`.

6. **Conclusions & recommendations**
   - Remove low-importance features to simplify the model (e.g., X6).
   - Prioritize operational activities that increase features strongly correlated with happiness.

---

## 🧾 Key Results (Summary)

- Dataset: 126 rows, 7 columns; class split — **69 happy, 57 unhappy**.
- Correlations observed: X1 & X5, X1 & X6, X3 & X5.
- Skewness: X1 highly negatively skewed; X5 & X6 moderately negatively skewed.
- Feature importance (Random Forest): **X6 was least important**.
- Best performing model (after removing X6): **Random Forest** —
  - **Accuracy ≈ 0.73**
  - **F1 Score ≈ 0.75**
- KNN and Logistic Regression produced lower performance on this dataset.

---

## 🛠 Reproducibility — Setup & Run

**Recommended:** use the provided conda environment.

```bash
# Create environment from shipped file
conda env create -f environment.yml
conda activate happycustomer

# OR using pip
conda create -n happycustomer python=3.10 -y
conda activate happycustomer
pip install -r setup/requirements.txt
```

**Run the analysis**
1. Place `ACME-HappinessSurvey2020.csv` inside the `data/` folder.
2. Open the notebook `src/HappyCustomer.ipynb` and run cells sequentially.
3. Generated outputs (figures and CSV summaries) will be saved to `reports/`.

---

## 🧠 Technical Decisions & Notes

- **Why Random Forest?** — Robust to small datasets, handles correlated features well, and provides feature importance measures that are easy to interpret for stakeholders.
- **Why remove X6?** — Empirical feature importance ranking showed X6 contributed least to predictive power; removing it simplified the model and improved performance slightly.
- **Imbalanced classes** — Class distribution is fairly balanced; standard metrics (accuracy, F1) were used. If imbalance grows, consider stratified sampling, class weighting, or resampling.

---

## 📣 Recommendations for Business Stakeholders

- Focus on improving the operational drivers that most strongly influence satisfaction (e.g., delivery timeliness and courier experience).
- Use the model to prioritize interventions (e.g., follow-up calls, targeted quality checks) for customers with low predicted satisfaction.
- Collect additional contextual features (e.g., delivery distance, order size, courier ID) to improve model predictive power.

---



**What I contributed in this project:**
- Performed end-to-end analysis: EDA, feature engineering, model development, and interpretability reporting.
- Produced reproducible artifacts (notebooks, reports, and visualizations) that are ready for stakeholder consumption.
- Made clear technical choices (model selection, preprocessing, feature reduction) and documented the reasoning and results.

**Skills demonstrated:**
- Data wrangling and EDA with pandas
- Statistical analysis (skewness, outliers, correlations)
- Supervised machine learning (scikit-learn): training, evaluation and interpretation
- Reproducible research: environment management and clear reporting
- Communication: clear visualizations and business-focused recommendations


