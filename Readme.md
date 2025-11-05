# Happy Customer Prediction

This project analyzes customer happiness survey data from ACME, a logistics and delivery startup, to predict whether customers are happy or unhappy based on various service-related features. The goal is to identify key factors influencing customer satisfaction and inform operational improvements to enhance customer experience.

## Problem Statement

The dataset contains feedback from customers about deliveries. We aim to predict customer happiness (target variable Y: 0 = unhappy, 1 = happy) using input features gathered from surveys. This analysis helps in taking necessary actions to improve services and ensure customer satisfaction.

## Data Description

The dataset is sourced from `ACME-HappinessSurvey2020.csv` and includes the following variables:

- **Y**: Target attribute (0 = unhappy, 1 = happy customers)
- **X1**: My order was delivered on time (ordinal scale)
- **X2**: Contents of my order was as I expected (ordinal scale)
- **X3**: I ordered everything I wanted to order (ordinal scale)
- **X4**: I paid a good price for my order (ordinal scale)
- **X5**: I am satisfied with my courier (ordinal scale)
- **X6**: The app makes ordering easy for me (ordinal scale)

All features are ordinal, typically ranging from 1 to 5, with no missing values in the dataset.

## Exploratory Data Analysis (EDA) Summary

The EDA performed in `src/main.ipynb` includes the following steps and insights:

### Data Overview
- Dataset loaded into a Pandas DataFrame.
- Basic info: 126 rows, 7 columns (X1-X6, Y).
- Descriptive statistics: Mean, median, mode calculated for each feature X1-X6 and saved to `reports/feature_stats.csv`.

### Missing Values
- No null values found in the dataset.

### Customer Distribution
- Bar plot showing counts of happy (Y=1) and unhappy (Y=0) customers.
- Happy customers: 69
- Unhappy customers: 57
- Plot saved as `reports/figures/customer_types.png`.

### Correlation Analysis
- Correlation matrix heatmap generated for all variables.
- Key observations:
  - X1 & X5 are closely related.
  - X1 & X6 are closely related.
  - X3 & X5 are closely related.
- Plot saved as `reports/figures/correlation_matrix.png`.

### Feature Distributions
- KDE plots for each feature (X1-X6) comparing distributions for happy vs. unhappy customers.
- Plots saved as `reports/figures/{feature}_distribution.png` for each feature.

### Outlier Detection
- Boxplots for each feature to visualize outliers.
- Outliers detected using IQR method for each feature.
- Outlier details merged with feature stats and saved to `reports/feature_stats_with_outliers.csv`.

### Skewness Analysis
- Skewness calculated for each feature:
  - X1: Highly negatively skewed
  - X2: Approximately symmetrical
  - X3: Approximately symmetrical
  - X4: Approximately symmetrical
  - X5: Moderately negatively skewed
  - X6: Moderately negatively skewed
- Skewness details merged with feature stats and saved to `reports/feature_stats_with_outliers_and_skewness.csv`.

### Conclusions from EDA
- Strong correlations between X1-X6, X3-X5, and X5-X6.
- Features X2, X3, X4 are symmetrical; X1 is highly negatively skewed; X5 and X6 are moderately negatively skewed.

## Modeling and Results

### Feature Importance
- RandomForestClassifier used to assess feature importance.
- Least important feature: X6.
- Plot saved as `reports/figures/feature_importance.png`.

### Model Training and Evaluation
- Models tested: RandomForestClassifier, LogisticRegression (with PowerTransformer and StandardScaler), KNeighborsClassifier.
- Train-test split: 80-20.
- After removing X6 (least important), RandomForest achieved: 
  - Accuracy: ~0.73
  - F1 Score: ~0.75
- KNN: 
  - Accuracy: ~0.65 
  - F1 Score: ~0.57.  
- LogisticRegression with preprocessing: Lower performance compared to others.


## Project Structure

```
happycustomer/
├── data/
│   └── ACME-HappinessSurvey2020.csv  # Original dataset
├── reports/
│   ├── feature_stats.csv  # Basic feature statistics
│   ├── feature_stats_with_outliers.csv  # Stats with outliers
│   ├── feature_stats_with_outliers_and_skewness.csv  # Stats with outliers and skewness
│   └── figures/  # Generated plots
│       ├── correlation_matrix.png
│       ├── customer_types.png
│       ├── feature_importance.png
│       └── X1_distribution.png to X6_distribution.png
├── setup/requirements.txt  # python dependencies
├── environment.yml         # Conda environment file
├── src/
│   └── HappyCustomer.ipynb  # Main analysis notebook with EDA and modeling
└── Readme.md  # This file
```

## Setup and Installation

1. Ensure Python 3.10 is installed.
2. Install required dependencies:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```
3. Clone or download the project.
4. Place the dataset in `data/` directory.

## Usage

- Open `src/main.ipynb` or `happycustomer.ipynb` in Jupyter Notebook.
- Run cells sequentially to reproduce EDA, visualizations, and modeling.
- Outputs (CSVs and PNGs) will be saved to `reports/` directory.

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

## Results and Insights

- Key predictors of happiness: Timely delivery (X1), courier satisfaction (X5), app ease (X6).
- Removing X6 improved model performance slightly.
- RandomForest performed best among tested models.
- Recommendations: Focus on improving delivery times(X1), satisfactory courier services(X5), and app usability to boost customer happiness (X6).

For further details, refer to the notebook (HappyCusotmer.ipynb) and generated reports.
