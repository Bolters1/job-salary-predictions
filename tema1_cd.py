"""
Tema 1 CD - Introducere in Machine Learning (VERSIUNE IMBUNATATITA)
Job Salary Prediction Dataset
- Clasificare: vacation (No Vacation, Small, Medium, Large)
- Regresie: salary

Modele folosite:
  Clasificare: Decision Tree, Logistic Regression, Random Forest, Gradient Boosting
  Regresie: Linear Regression, Ridge (L2), Lasso (L1), Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ============================================================================
# PASUL 1: INCARCAREA DATELOR
# ============================================================================
print("=" * 70)
print("PASUL 1: INCARCAREA DATELOR")
print("=" * 70)

df_train_full = pd.read_csv('CD_education_economy.csv')
df_test = pd.read_csv('CD_education_economy_test.csv')

print(f"Train set: {df_train_full.shape[0]} randuri, {df_train_full.shape[1]} coloane")
print(f"Test set:  {df_test.shape[0]} randuri, {df_test.shape[1]} coloane")
print(f"\nColoane: {list(df_train_full.columns)}")

# ============================================================================
# PASUL 2: EDA - EXPLORATORY DATA ANALYSIS (2p)
# ============================================================================
print("\n" + "=" * 70)
print("PASUL 2: EDA - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

target_cls = 'vacation'
target_reg = 'salary'

numeric_cols = df_train_full.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_train_full.select_dtypes(include=['object']).columns.tolist()

print(f"\nColoane numerice: {numeric_cols}")
print(f"Coloane categoriale: {categorical_cols}")

# Statistici numerice
print("\n--- Statistici Atribute Numerice ---")
numeric_stats = df_train_full[numeric_cols].describe().T
numeric_stats['missing'] = df_train_full[numeric_cols].isnull().sum()
numeric_stats['non_null'] = df_train_full[numeric_cols].notnull().sum()
print(numeric_stats[['non_null', 'missing', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']])

# Statistici categoriale
print("\n--- Statistici Atribute Categoriale ---")
cat_stats = pd.DataFrame({
    'non_null': df_train_full[categorical_cols].notnull().sum(),
    'missing': df_train_full[categorical_cols].isnull().sum(),
    'unique': df_train_full[categorical_cols].nunique(),
    'top_value': [df_train_full[c].mode()[0] for c in categorical_cols],
    'top_freq': [df_train_full[c].value_counts().iloc[0] for c in categorical_cols]
})
print(cat_stats)

# --- GRAFICE EDA ---

# GRAFIC 1: Boxploturi numerice
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Boxplot-uri Atribute Numerice', fontsize=16, fontweight='bold')
for i, col in enumerate(numeric_cols[:6]):
    ax = axes[i // 3][i % 3]
    sns.boxplot(data=df_train_full, y=col, ax=ax, color='steelblue')
    ax.set_title(col, fontsize=12)
plt.tight_layout()
plt.savefig('plots/01_boxplots_numeric.png', bbox_inches='tight')
plt.close()
print("\n[SAVED] plots/01_boxplots_numeric.png")

# GRAFIC 2: Histograme categoriale
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Distributia Atributelor Categoriale', fontsize=16, fontweight='bold')
for i, col in enumerate(categorical_cols[:6]):
    ax = axes[i // 3][i % 3]
    order = df_train_full[col].value_counts().index
    sns.countplot(data=df_train_full, x=col, ax=ax, order=order, palette='Set2')
    ax.set_title(col, fontsize=12)
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('plots/02_histograms_categorical.png', bbox_inches='tight')
plt.close()
print("[SAVED] plots/02_histograms_categorical.png")

# GRAFIC 3: Certifications missing analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Analiza Coloanei certifications (valori lipsa)', fontsize=14, fontweight='bold')
missing_count = df_train_full['certifications'].isnull().sum()
non_missing_count = df_train_full['certifications'].notnull().sum()
axes[0].pie([non_missing_count, missing_count], labels=['Prezent', 'Lipsa'],
            autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
axes[0].set_title('Valori lipsa in certifications')
df_train_full['certifications'].dropna().hist(ax=axes[1], bins=20, color='steelblue', edgecolor='black')
axes[1].set_title('Distributia certifications (fara NaN)')
plt.tight_layout()
plt.savefig('plots/03_certifications_missing.png', bbox_inches='tight')
plt.close()
print("[SAVED] plots/03_certifications_missing.png")

# GRAFIC 4: Class balance
print("\n--- Echilibrul claselor (vacation) ---")
class_distribution = df_train_full[target_cls].value_counts()
print(class_distribution)
print(f"\nProcentaje:")
print((class_distribution / len(df_train_full) * 100).round(2))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Echilibrul Claselor - vacation', fontsize=16, fontweight='bold')
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
class_distribution.plot(kind='bar', ax=axes[0], color=colors, edgecolor='black')
axes[0].set_title('Frecventa fiecarei clase')
axes[0].set_ylabel('Numar exemple')
axes[0].tick_params(axis='x', rotation=0)
for i, (idx, val) in enumerate(class_distribution.items()):
    axes[0].text(i, val + 200, str(val), ha='center', fontweight='bold')
axes[1].pie(class_distribution.values, labels=class_distribution.index,
            autopct='%1.1f%%', colors=colors, startangle=90)
axes[1].set_title('Proportia fiecarei clase')
plt.tight_layout()
plt.savefig('plots/04_class_balance_vacation.png', bbox_inches='tight')
plt.close()
print("[SAVED] plots/04_class_balance_vacation.png")

# GRAFIC 5: Heatmap correlatie
corr_matrix = df_train_full[numeric_cols].corr()
print("\nMatrice de corelatie (numerice):")
print(corr_matrix.round(3))

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            square=True, ax=ax, vmin=-1, vmax=1)
ax.set_title('Heatmap Corelatie - Atribute Numerice', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/05_correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("[SAVED] plots/05_correlation_heatmap.png")

# Chi-Square test intre atribute categorice
print("\n--- Chi-Square: Corelatie intre atribute categorice ---")
cat_cols_no_target = [c for c in categorical_cols if c != target_cls]
chi2_results = []
for i in range(len(cat_cols_no_target)):
    for j in range(i+1, len(cat_cols_no_target)):
        ct = pd.crosstab(df_train_full[cat_cols_no_target[i]], df_train_full[cat_cols_no_target[j]])
        chi2, p, dof, expected = chi2_contingency(ct)
        chi2_results.append({
            'Atribut 1': cat_cols_no_target[i],
            'Atribut 2': cat_cols_no_target[j],
            'Chi2': round(chi2, 2),
            'p-value': round(p, 6),
            'Corelate': 'Da' if p < 0.05 else 'Nu'
        })
chi2_df = pd.DataFrame(chi2_results)
print(chi2_df.to_string(index=False))
corelate = chi2_df[chi2_df['Corelate'] == 'Da']
print(f"\nPerechi corelate semnificativ: {len(corelate)} din {len(chi2_df)}")

# GRAFIC 6: Numeric vs Classification
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df_train_full, x=target_cls, y='experience_years', ax=ax,
            order=['No Vacation', 'Small', 'Medium', 'Large'], palette='Set2')
ax.set_title('experience_years vs vacation (Numeric vs Clasificare)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/06_experience_vs_vacation.png', bbox_inches='tight')
plt.close()
print("[SAVED] plots/06_experience_vs_vacation.png")

# GRAFIC 7: Categorical vs Classification
fig, ax = plt.subplots(figsize=(10, 6))
ct = pd.crosstab(df_train_full['education_level'], df_train_full[target_cls], normalize='index')
ct.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
ax.set_title('education_level vs vacation (Categorial vs Clasificare)', fontsize=14, fontweight='bold')
ax.set_ylabel('Proportie')
ax.legend(title='vacation')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('plots/07_education_vs_vacation.png', bbox_inches='tight')
plt.close()
print("[SAVED] plots/07_education_vs_vacation.png")

# GRAFIC 8: Numeric vs Regression
fig, ax = plt.subplots(figsize=(10, 6))
sample_idx = df_train_full.sample(min(5000, len(df_train_full)), random_state=42).index
sns.scatterplot(data=df_train_full.loc[sample_idx], x='experience_years', y=target_reg,
                alpha=0.3, ax=ax, color='steelblue', s=10)
ax.set_title('experience_years vs salary (Numeric vs Regresie)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/08_experience_vs_salary.png', bbox_inches='tight')
plt.close()
print("[SAVED] plots/08_experience_vs_salary.png")

# GRAFIC 9: Categorical vs Regression
fig, ax = plt.subplots(figsize=(12, 6))
order = df_train_full.groupby('location')[target_reg].median().sort_values(ascending=False).index
sns.boxplot(data=df_train_full, x='location', y=target_reg, ax=ax, order=order, palette='Set3')
ax.set_title('location vs salary (Categorial vs Regresie)', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('plots/09_location_vs_salary.png', bbox_inches='tight')
plt.close()
print("[SAVED] plots/09_location_vs_salary.png")

# GRAFIC 10: Salary distribution
fig, ax = plt.subplots(figsize=(10, 6))
df_train_full[target_reg].hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(df_train_full[target_reg].mean(), color='red', linestyle='--',
           label=f'Mean: {df_train_full[target_reg].mean():.0f}')
ax.axvline(df_train_full[target_reg].median(), color='green', linestyle='--',
           label=f'Median: {df_train_full[target_reg].median():.0f}')
ax.set_title('Distributia Salary', fontsize=14, fontweight='bold')
ax.set_xlabel('Salary')
ax.legend()
plt.tight_layout()
plt.savefig('plots/10_salary_distribution.png', bbox_inches='tight')
plt.close()
print("[SAVED] plots/10_salary_distribution.png")

print("\n>>> EDA COMPLET! 10 grafice salvate in /plots/")

# ============================================================================
# PASUL 3: PREPROCESARE DATE (2p)
# ============================================================================
print("\n" + "=" * 70)
print("PASUL 3: PREPROCESARE DATE")
print("=" * 70)

# --- 3.1 Imputare ---
print("\n--- 3.1 Imputare date lipsa ---")
print("Valori lipsa inainte:")
print(df_train_full.isnull().sum()[df_train_full.isnull().sum() > 0])

cert_median = df_train_full['certifications'].median()
print(f"\nImput 'certifications' cu mediana = {cert_median}")
df_train_full['certifications'] = df_train_full['certifications'].fillna(cert_median)
df_test['certifications'] = df_test['certifications'].fillna(cert_median)

# --- 3.2 Outlieri ---
print("\n--- 3.2 Tratare outlieri ---")

def detect_and_cap_outliers(df, col, factor=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    if n_outliers > 0:
        print(f"  {col}: {n_outliers} outlieri (range: [{lower:.2f}, {upper:.2f}])")
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df

outlier_cols = ['experience_years', 'skills_count', 'certifications', 'proficiencies', 'aggregated_score']
for col in outlier_cols:
    df_train_full = detect_and_cap_outliers(df_train_full, col)
    df_test = detect_and_cap_outliers(df_test, col)

# --- 3.3 Feature Engineering ---
print("\n--- 3.3 Feature Engineering ---")

for df in [df_train_full, df_test]:
    df['exp_x_skills'] = df['experience_years'] * df['skills_count']
    df['exp_x_certs'] = df['experience_years'] * df['certifications']
    df['skills_x_certs'] = df['skills_count'] * df['certifications']
    df['exp_per_skill'] = df['experience_years'] / (df['skills_count'] + 1)
    df['cert_per_exp'] = df['certifications'] / (df['experience_years'] + 1)

edu_order = {'High School': 0, 'Diploma': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
size_order = {'Startup': 0, 'Small': 1, 'Medium': 2, 'Large': 3, 'Enterprise': 4}
for df in [df_train_full, df_test]:
    df['education_ordinal'] = df['education_level'].map(edu_order)
    df['company_size_ordinal'] = df['company_size'].map(size_order)

print("  Features noi create: exp_x_skills, exp_x_certs, skills_x_certs, exp_per_skill, cert_per_exp")
print("  Encoding ordinal: education_ordinal, company_size_ordinal")

# --- 3.4 Encoding ---
print("\n--- 3.4 Encoding ---")

le_vacation = LabelEncoder()
df_train_full['vacation_encoded'] = le_vacation.fit_transform(df_train_full[target_cls])
print(f"Clase vacation: {dict(zip(le_vacation.classes_, le_vacation.transform(le_vacation.classes_)))}")

y_cls_full = df_train_full['vacation_encoded'].values
y_reg_full = df_train_full[target_reg].values

feature_cols = [c for c in df_train_full.columns if c not in [target_cls, target_reg, 'vacation_encoded']]

df_features_train = pd.get_dummies(df_train_full[feature_cols], drop_first=True)
df_features_test = pd.get_dummies(df_test[feature_cols], drop_first=True)
df_features_test = df_features_test.reindex(columns=df_features_train.columns, fill_value=0)

df_features_train = df_features_train.fillna(0)
df_features_test = df_features_test.fillna(0)

print(f"Features dupa encoding: {df_features_train.shape[1]} coloane")

# --- 3.4 Standardizare ---
print("\n--- 3.4 Standardizare ---")
scaler = StandardScaler()
X_full = scaler.fit_transform(df_features_train)
X_test_final = scaler.transform(df_features_test)

# --- 3.5 Split ---
print("\n--- 3.5 Split Train/Validare ---")
X_train, X_val, y_train_cls, y_val_cls, y_train_reg, y_val_reg = train_test_split(
    X_full, y_cls_full, y_reg_full, test_size=0.2, random_state=42, stratify=y_cls_full
)
print(f"Train: {X_train.shape[0]} | Validare: {X_val.shape[0]}")

print("\n>>> PREPROCESARE COMPLETA!")

# ============================================================================
# PASUL 4: CLASIFICARE (3p)
# ============================================================================
print("\n" + "=" * 70)
print("PASUL 4: CLASIFICARE - Predictia 'vacation'")
print("=" * 70)

def evaluate_clf(model, X_tr, y_tr, X_v, y_v, name):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_v)
    acc = accuracy_score(y_v, y_pred)
    prec = precision_score(y_v, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_v, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_v, y_pred, average='weighted', zero_division=0)
    print(f"  {name:45s} | Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}")
    return {'name': name, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'model': model}

all_cls = []

# --- 4.1 Decision Tree baseline ---
print("\n--- 4.1 Decision Tree Classifier ---")
print(f"{'Model':45s} | {'Acc':>8s} | {'Prec':>8s} | {'Rec':>8s} | {'F1':>8s}")
print("-" * 95)
for depth in [3, 5, 10, 15]:
    r = evaluate_clf(DecisionTreeClassifier(max_depth=depth, random_state=42),
                     X_train, y_train_cls, X_val, y_val_cls, f"DT(depth={depth})")
    all_cls.append(r)

# Best depth ablation with min_samples_leaf
best_dt = max(all_cls, key=lambda x: x['f1'])
bd = best_dt['model'].max_depth
for msl in [5, 10, 20, 50]:
    r = evaluate_clf(DecisionTreeClassifier(max_depth=bd, min_samples_leaf=msl, random_state=42),
                     X_train, y_train_cls, X_val, y_val_cls, f"DT(depth={bd}, msl={msl})")
    all_cls.append(r)

# --- 4.2 Logistic Regression ---
print("\n--- 4.2 Logistic Regression ---")
print(f"{'Model':45s} | {'Acc':>8s} | {'Prec':>8s} | {'Rec':>8s} | {'F1':>8s}")
print("-" * 95)
for C in [0.01, 0.1, 1, 10, 100]:
    r = evaluate_clf(LogisticRegression(C=C, max_iter=2000, random_state=42, solver='lbfgs'),
                     X_train, y_train_cls, X_val, y_val_cls, f"LogReg(C={C})")
    all_cls.append(r)

# --- 4.3 Random Forest (UPGRADE!) ---
print("\n--- 4.3 Random Forest Classifier ---")
print(f"{'Model':45s} | {'Acc':>8s} | {'Prec':>8s} | {'Rec':>8s} | {'F1':>8s}")
print("-" * 95)
for n_est in [50, 100, 200]:
    for depth in [10, 15, 20]:
        r = evaluate_clf(
            RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1),
            X_train, y_train_cls, X_val, y_val_cls,
            f"RF(n={n_est}, depth={depth})")
        all_cls.append(r)

# Best RF with min_samples_leaf tuning
best_rf_cls = max([r for r in all_cls if 'RF(' in r['name']], key=lambda x: x['f1'])
rf_params = best_rf_cls['model']
print(f"\nBest RF so far: {best_rf_cls['name']} (F1={best_rf_cls['f1']:.4f})")
print("Tuning min_samples_leaf pe best RF...")
for msl in [1, 2, 5, 10]:
    r = evaluate_clf(
        RandomForestClassifier(n_estimators=rf_params.n_estimators, max_depth=rf_params.max_depth,
                               min_samples_leaf=msl, random_state=42, n_jobs=-1),
        X_train, y_train_cls, X_val, y_val_cls,
        f"RF(n={rf_params.n_estimators}, d={rf_params.max_depth}, msl={msl})")
    all_cls.append(r)

# --- 4.4 Gradient Boosting Classifier ---
print("\n--- 4.4 Gradient Boosting Classifier ---")
print(f"{'Model':45s} | {'Acc':>8s} | {'Prec':>8s} | {'Rec':>8s} | {'F1':>8s}")
print("-" * 95)
for max_iter in [100, 200, 300]:
    for lr in [0.05, 0.1]:
        r = evaluate_clf(
            HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=lr,
                                       max_depth=5, random_state=42),
            X_train, y_train_cls, X_val, y_val_cls,
            f"GB(iter={max_iter}, lr={lr}, depth=5)")
        all_cls.append(r)

best_gb_cls = max([r for r in all_cls if 'GB(' in r['name']], key=lambda x: x['f1'])
print(f"\nBest GB: {best_gb_cls['name']} (F1={best_gb_cls['f1']:.4f})")
print("Tuning max_depth on best GB...")
for d in [3, 5, 7]:
    r = evaluate_clf(
        HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                   max_depth=d, random_state=42),
        X_train, y_train_cls, X_val, y_val_cls,
        f"GB(iter=200, lr=0.1, depth={d})")
    all_cls.append(r)

# --- Tabel sumar clasificare ---
cls_df = pd.DataFrame([{
    'Model': r['name'], 'Accuracy': r['acc'], 'Precision': r['prec'],
    'Recall': r['rec'], 'F1-Score': r['f1']
} for r in all_cls]).sort_values('F1-Score', ascending=False)

print("\n" + "=" * 70)
print("TABEL SUMAR - CLASIFICARE (vacation) - Top 10")
print("=" * 70)
print(cls_df.head(10).to_string(index=False, float_format='%.4f'))
cls_df.to_csv('results/classification_results.csv', index=False)

best_cls = max(all_cls, key=lambda x: x['f1'])
print(f"\n>>> BEST CLASIFICARE: {best_cls['name']} (F1={best_cls['f1']:.4f})")

# Confusion matrix
best_cls_model = best_cls['model']
y_pred_best = best_cls_model.predict(X_val)
cm = confusion_matrix(y_val_cls, y_pred_best)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=le_vacation.classes_, yticklabels=le_vacation.classes_)
ax.set_title(f'Confusion Matrix - {best_cls["name"]}', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('plots/11_confusion_matrix_best.png', bbox_inches='tight')
plt.close()
print("[SAVED] plots/11_confusion_matrix_best.png")

# Full classification report for best model
print(f"\nClassification Report - {best_cls['name']}:")
print(classification_report(y_val_cls, y_pred_best, target_names=le_vacation.classes_, zero_division=0))

# ============================================================================
# PASUL 5: REGRESIE (3p)
# ============================================================================
print("\n" + "=" * 70)
print("PASUL 5: REGRESIE - Predictia 'salary'")
print("=" * 70)

def evaluate_reg(model, X_tr, y_tr, X_v, y_v, name):
    model.fit(X_tr, y_tr)
    y_pred_tr = model.predict(X_tr)
    y_pred_v = model.predict(X_v)
    r2_tr = r2_score(y_tr, y_pred_tr)
    r2_v = r2_score(y_v, y_pred_v)
    rmse_tr = np.sqrt(mean_squared_error(y_tr, y_pred_tr))
    rmse = np.sqrt(mean_squared_error(y_v, y_pred_v))
    mae = mean_absolute_error(y_v, y_pred_v)
    print(f"  {name:45s} | R2_train={r2_tr:.4f} | R2_val={r2_v:.4f} | RMSE={rmse:.0f} | MAE={mae:.0f}")
    return {'name': name, 'r2_train': r2_tr, 'r2': r2_v, 'rmse_train': rmse_tr, 'rmse': rmse, 'mae': mae, 'model': model}

all_reg = []

# --- 5.1 Linear Regression baseline ---
print("\n--- 5.1 Linear Regression (baseline) ---")
print(f"{'Model':45s} | {'R2_train':>10s} | {'R2_val':>10s} | {'RMSE':>8s} | {'MAE':>8s}")
print("-" * 100)
r = evaluate_reg(LinearRegression(), X_train, y_train_reg, X_val, y_val_reg, "Linear Regression")
all_reg.append(r)

# --- 5.2 Ridge Regression (L2 regularization) ---
print("\n--- 5.2 Ridge Regression (L2 regularization) ---")
print(f"{'Model':45s} | {'R2_train':>10s} | {'R2_val':>10s} | {'RMSE':>8s} | {'MAE':>8s}")
print("-" * 100)
for alpha in [0.01, 0.1, 1, 10, 100, 1000]:
    r = evaluate_reg(Ridge(alpha=alpha), X_train, y_train_reg, X_val, y_val_reg,
                     f"Ridge(alpha={alpha})")
    all_reg.append(r)

# --- 5.3 Lasso Regression (L1 regularization) ---
print("\n--- 5.3 Lasso Regression (L1 regularization) ---")
print(f"{'Model':45s} | {'R2_train':>10s} | {'R2_val':>10s} | {'RMSE':>8s} | {'MAE':>8s}")
print("-" * 100)
for alpha in [0.01, 0.1, 1, 10, 100]:
    r = evaluate_reg(Lasso(alpha=alpha, max_iter=5000), X_train, y_train_reg, X_val, y_val_reg,
                     f"Lasso(alpha={alpha})")
    all_reg.append(r)

# --- 5.4 Decision Tree Regressor ---
print("\n--- 5.4 Decision Tree Regressor ---")
print(f"{'Model':45s} | {'R2_train':>10s} | {'R2_val':>10s} | {'RMSE':>8s} | {'MAE':>8s}")
print("-" * 100)
for depth in [5, 10, 15, 20]:
    r = evaluate_reg(DecisionTreeRegressor(max_depth=depth, random_state=42),
                     X_train, y_train_reg, X_val, y_val_reg, f"DTR(depth={depth})")
    all_reg.append(r)

# --- 5.5 Random Forest Regressor (UPGRADE!) ---
print("\n--- 5.5 Random Forest Regressor ---")
print(f"{'Model':45s} | {'R2_train':>10s} | {'R2_val':>10s} | {'RMSE':>8s} | {'MAE':>8s}")
print("-" * 100)
for n_est in [50, 100, 200]:
    for depth in [10, 15, 20]:
        r = evaluate_reg(
            RandomForestRegressor(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1),
            X_train, y_train_reg, X_val, y_val_reg,
            f"RFR(n={n_est}, depth={depth})")
        all_reg.append(r)

# Best RFR tuning
best_rfr = max([r for r in all_reg if 'RFR(' in r['name']], key=lambda x: x['r2'])
rfr_p = best_rfr['model']
print(f"\nBest RFR so far: {best_rfr['name']} (R2={best_rfr['r2']:.4f})")
print("Tuning min_samples_leaf pe best RFR...")
for msl in [1, 2, 5, 10]:
    r = evaluate_reg(
        RandomForestRegressor(n_estimators=rfr_p.n_estimators, max_depth=rfr_p.max_depth,
                              min_samples_leaf=msl, random_state=42, n_jobs=-1),
        X_train, y_train_reg, X_val, y_val_reg,
        f"RFR(n={rfr_p.n_estimators}, d={rfr_p.max_depth}, msl={msl})")
    all_reg.append(r)

# --- 5.6 Gradient Boosting Regressor ---
print("\n--- 5.6 Gradient Boosting Regressor ---")
print(f"{'Model':45s} | {'R2_train':>10s} | {'R2_val':>10s} | {'RMSE':>8s} | {'MAE':>8s}")
print("-" * 100)
for max_iter in [100, 200, 300]:
    for lr in [0.05, 0.1]:
        r = evaluate_reg(
            HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=lr,
                                      max_depth=5, random_state=42),
            X_train, y_train_reg, X_val, y_val_reg,
            f"GBR(iter={max_iter}, lr={lr}, depth=5)")
        all_reg.append(r)

print("Tuning max_depth on GBR...")
for d in [3, 5, 7]:
    r = evaluate_reg(
        HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1,
                                  max_depth=d, random_state=42),
        X_train, y_train_reg, X_val, y_val_reg,
        f"GBR(iter=300, lr=0.1, depth={d})")
    all_reg.append(r)

# --- Tabel sumar regresie ---
reg_df = pd.DataFrame([{
    'Model': r['name'], 'R2_train': r['r2_train'], 'R2_val': r['r2'],
    'RMSE': r['rmse'], 'MAE': r['mae']
} for r in all_reg]).sort_values('R2_val', ascending=False)

print("\n" + "=" * 70)
print("TABEL SUMAR - REGRESIE (salary) - Top 10")
print("=" * 70)
print(reg_df.head(10).to_string(index=False, float_format='%.4f'))
reg_df.to_csv('results/regression_results.csv', index=False)

best_reg = max(all_reg, key=lambda x: x['r2'])
print(f"\n>>> BEST REGRESIE: {best_reg['name']} (R2={best_reg['r2']:.4f}, RMSE={best_reg['rmse']:.0f})")

# GRAFIC 12: Predicted vs Actual
best_reg_model = best_reg['model']
y_pred_reg_val = best_reg_model.predict(X_val)
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_val_reg, y_pred_reg_val, alpha=0.2, s=5, color='steelblue')
lims = [min(y_val_reg.min(), y_pred_reg_val.min()), max(y_val_reg.max(), y_pred_reg_val.max())]
ax.plot(lims, lims, 'r--', alpha=0.8, label='Perfect prediction')
ax.set_xlabel('Actual Salary'); ax.set_ylabel('Predicted Salary')
ax.set_title(f'Predicted vs Actual - {best_reg["name"]} (R²={best_reg["r2"]:.4f})',
             fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('plots/12_predicted_vs_actual_salary.png', bbox_inches='tight')
plt.close()
print("[SAVED] plots/12_predicted_vs_actual_salary.png")

# GRAFIC 13: Residuals
residuals = y_val_reg - y_pred_reg_val
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_pred_reg_val, residuals, alpha=0.2, s=5, color='steelblue')
ax.axhline(y=0, color='red', linestyle='--')
ax.set_xlabel('Predicted Salary'); ax.set_ylabel('Residual')
ax.set_title(f'Residuals Plot - {best_reg["name"]}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/13_residuals_plot.png', bbox_inches='tight')
plt.close()
print("[SAVED] plots/13_residuals_plot.png")

# GRAFIC 14: Overfitting Check (Train vs Val)
print("\n--- Generare curbe train vs val (Line Plots) ---")
dtr_res = [r for r in all_reg if r['name'].startswith('DTR(depth=')]
depths = [int(r['name'].split('=')[1].replace(')', '')) for r in dtr_res]
mse_train = [r['rmse_train']**2 for r in dtr_res]
mse_val = [r['rmse']**2 for r in dtr_res]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(depths, mse_train, label='Train MSE w/ reg', marker='o')
ax.plot(depths, mse_val, label='Test MSE w/ reg', marker='o')
ax.set_xlabel('max_depth (M)')
ax.set_ylabel('MSE')
ax.set_title('Comportament comparativ pe setul de antrenare versus cel de validare', fontsize=12, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('plots/14_overfitting_curve.png', bbox_inches='tight')
plt.close()
print("[SAVED] plots/14_overfitting_curve.png")

# GRAFIC 15: Comparative Training
rf100_res = [r for r in all_reg if r['name'].startswith('RFR(n=100')]
rf200_res = [r for r in all_reg if r['name'].startswith('RFR(n=200')]
d_rf100 = [int(r['name'].split('depth=')[1].replace(')', '')) for r in rf100_res]
mse_tr100 = [r['rmse_train']**2 for r in rf100_res]
d_rf200 = [int(r['name'].split('depth=')[1].replace(')', '')) for r in rf200_res]
mse_tr200 = [r['rmse_train']**2 for r in rf200_res]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(depths, mse_train, label='DTR Training MSE', marker='o')
ax.plot(d_rf100, mse_tr100, label='RF(n=100) Training MSE', marker='o')
ax.plot(d_rf200, mse_tr200, label='RF(n=200) Training MSE', marker='o')
ax.set_xlabel('max_depth (M)')
ax.set_ylabel('MSE (loss)')
ax.set_title('Evolutia comparativa pe setul de antrenare a mai multor experimente', fontsize=12, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('plots/15_comparative_training.png', bbox_inches='tight')
plt.close()
print("[SAVED] plots/15_comparative_training.png")

# ============================================================================
# PASUL 6: PREDICTII PE SETUL DE TEST
# ============================================================================
print("\n" + "=" * 70)
print("PASUL 6: PREDICTII PE SETUL DE TEST")
print("=" * 70)

# Use best models
test_vacation_pred = best_cls_model.predict(X_test_final)
test_vacation_labels = le_vacation.inverse_transform(test_vacation_pred)
test_salary_pred = best_reg_model.predict(X_test_final)

predictions_df = pd.DataFrame({
    'vacation_predicted': test_vacation_labels,
    'salary_predicted': test_salary_pred
})
predictions_df.to_csv('results/test_predictions.csv', index=False)

# Salvare predictii
predictions_df = pd.DataFrame({
    'vacation_predicted': test_vacation_labels,
    'salary_predicted': test_salary_pred
})
predictions_df.to_csv('results/test_predictions.csv', index=False)

print(f"\nPredictii salvate in results/test_predictions.csv")
print(f"\nVacation distribution:")
print(pd.Series(test_vacation_labels).value_counts())
print(f"\nSalary: Mean={test_salary_pred.mean():.0f}, Std={test_salary_pred.std():.0f}")

# ============================================================================
# SUMAR FINAL
# ============================================================================
print("\n" + "=" * 70)
print("SUMAR FINAL")
print("=" * 70)
print(f"""
CLASIFICARE (vacation):
  Best: {best_cls['name']}
  F1={best_cls['f1']:.4f} | Acc={best_cls['acc']:.4f} | Prec={best_cls['prec']:.4f}

REGRESIE (salary):
  Best: {best_reg['name']}
  R²={best_reg['r2']:.4f} | RMSE={best_reg['rmse']:.0f} | MAE={best_reg['mae']:.0f}

Fisiere: plots/ (14 grafice), results/classification_results.csv, results/regression_results.csv, results/test_predictions.csv
""")
print("DONE!")
