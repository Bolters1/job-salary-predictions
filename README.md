# Job Salary & Vacation Prediction

Machine learning project tackling two tasks on a job/salary dataset (~64k samples, 14 features):

- **Classification** — predicting vacation tier (No Vacation / Small / Medium / Large)
- **Regression** — predicting salary

## Models

| Task | Models |
|------|--------|
| Classification | Decision Tree, Logistic Regression, Random Forest, Gradient Boosting |
| Regression | Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting |

## Structure

```
tema1_cd.ipynb                 # main notebook — full pipeline end-to-end
CD_education_economy.csv       # training data (64k rows)
CD_education_economy_test.csv  # held-out test data (16k rows)
plots/               # generated visualizations (EDA, confusion matrices, residuals, etc.)
results/             # prediction CSVs and metric summaries
```

## Highlights

- Full EDA: boxplots, histograms, correlation heatmap, class balance analysis
- Handles ~30% missing values in `certifications` via median imputation
- Addresses class imbalance with `class_weight='balanced'`
- Hyperparameter tuning and overfitting analysis with train/validation curves
- Comparative evaluation across all models with detailed metrics

## Running

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
jupyter notebook tema1_cd.ipynb
```

Plots are saved to `plots/`, results to `results/`.
