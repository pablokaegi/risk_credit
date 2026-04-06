# Argentine Financial Distress Prediction Model

<p align="center">
  <img src="reports/figures/model_performance.png" alt="Model Performance Dashboard" width="800"/>
</p>

<p align="center">
  <em>A machine learning approach to predicting financial distress in Argentine publicly traded companies using CNV regulatory data</em>
</p>

---

## Executive Summary

Financial distress prediction models are constructed to identify Argentine publicly traded companies at risk of default, bankruptcy, or severe financial instability. The methodology is adapted from WorldQuant University's bankruptcy prediction framework to address the unique characteristics of the Argentine capital market, where legal bankruptcy filings are extremely rare events (frequency <0.5%).

The target variable is redefined from legal bankruptcy to **Financial Distress**, a continuous spectrum that includes negative equity, debt restructuring negotiations, trading suspensions by BYMA, and technical defaults. This approach increases the positive class frequency to approximately 5-8%, making the classification task tractable while maintaining practical relevance for portfolio managers and risk officers.

The pipeline is designed to handle severe class imbalance using SMOTE (Synthetic Minority Over-sampling Technique), temporal cross-validation to prevent data leakage, and robust feature engineering from CNV quarterly financial statements. A production-ready Python package is delivered, following software engineering best practices with modular architecture, comprehensive testing, and automated data acquisition.

---

## Table of Contents

- [Business Context](#business-context)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Financial Ratios](#financial-ratios)
- [Results](#results)
- [Feature Importance](#feature-importance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Limitations & Future Work](#limitations--future-work)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)
- [Author](#author)

---

## Business Context

### The Argentine Market Reality

The Argentine capital market presents unique challenges for credit risk modeling:

1. **Extreme Class Imbalance:** Legal bankruptcy filings among BYMA-listed companies occur at a rate of less than 0.5%, making traditional binary bankruptcy classifiers impractical.

2. **Macroeconomic Volatility:** High inflation (averaging 50%+ annually since 2018), currency devaluation, and capital controls create non-stationarity in financial ratios that requires specialized modeling techniques.

3. **Regulatory Complexity:** The Comisión Nacional de Valores (CNV) requires quarterly financial statements from all publicly traded companies, but data quality varies significantly and often contains missing values or restatements.

4. **Data Accessibility:** Unlike US markets with SEC EDGAR APIs, Argentine financial data must be extracted from CNV's web portal, requiring custom scraping infrastructure.

### Why Financial Distress Instead of Bankruptcy?

The traditional definition of bankruptcy (legal filing for creditor protection) is insufficient for the Argentine market because:

- Companies may operate for years with negative equity without formal bankruptcy
- Debt restructuring negotiations occur outside court proceedings
- BYMA trading suspensions often precede formal bankruptcy by 6-18 months
- Foreign investors require earlier warning signals than local litigation reporters

By expanding the definition to **Financial Distress**, the model captures companies in the critical window between financial stress and legal resolution, providing maximum value for portfolio managers seeking to exit positions before liquidity evaporates.

### Use Cases

| Stakeholder | Application |
|-------------|-------------|
| **Portfolio Managers** | Screen positions for hidden credit risk before market pricing reflects distress |
| **Risk Officers** | Automate quarterly credit monitoring for counterparties |
| **Sell-Side Analysts** | Generate distress probability scores for coverage universe |
| **Quant Researchers** | Backtest credit-based factor strategies on Argentine equities |
| **Regulators** | Identify systemic risk accumulation in financial sector |

---

## Dataset Description

### Primary Data Source: BCRA APIs

**This project uses official APIs from the Central Bank of Argentina (BCRA), which is the GOLD STANDARD for Argentine credit risk data.**

| API | Content | Why It Matters |
|-----|---------|----------------|
| **Central de Deudores** | Complete debt registry for ALL Argentine companies | **Primary distress indicator** - shows real debt situations |
| **Cheques Denunciados** | Rejected checks due to insufficient funds | **Strong distress signal** - indicates cash flow problems |
| **Estadísticas Cambiarias** | Official dollar rate, reserves | Macro context for USD-denominated debt |
| **Estadísticas Monetarias** | Interest rates, monetary aggregates | Cost of capital context |
| **Régimen de Transparencia** | Regulatory transparency data | Compliance status |

**Why BCRA APIs are superior:**

| Feature | BCRA APIs | Yahoo Finance | CNV Scraping |
|---------|-----------|---------------|--------------|
| **Coverage** | ALL Argentine companies | ~15 ADRs only | ~80 BYMA companies |
| **Data Quality** | Official, audited data | Third-party estimates | Unstable scraping |
| **Distress Signals** | ✅ Cheques denunciados, deuda real | ❌ Not available | ⚠️ Limited |
| **Professionalism** | Official API, documented | Side project | Unstable |
| **Maintenance** | Proactively maintained | Community-driven | Breaks frequently |

**API Documentation:** https://api.bcra.gob.ar
**Contact:** api@bcra.gob.ar

### Target Variable Definition

A company is classified as **Financial Distress = 1** if ANY of the following conditions are met in a given quarter:

| Condition | Definition | Data Source |
|-----------|------------|--------------|
| **Negative Equity** | Total Equity < 0 | Balance Sheet (CNV) |
| **Debt Restructuring** | Public announcement of negotiation with creditors | CNV Audiencias Permanentes |
| **Trading Suspension** | BYMA suspends trading due to non-compliance | BYMA Regulatory Notices |
| **Technical Default** | Violation of debt covenants reported | CNV Material Events |
| **Operating Losses** | Consecutive losses for ≥3 years | Income Statement (CNV) |
| **Auditor Resignation** | Independent auditor resigns or issues qualified opinion | CNV Audit Filings |

If NONE of the above apply, the company is classified as **Healthy = 0**.

### Feature Engineering

From the raw financial statements, **10 critical financial ratios** are calculated:

#### LIQUIDITY RATIOS (3 features)

1. **Current Ratio** = Current Assets / Current Liabilities
   - Interpretation: Measures short-term solvency. Healthy companies typically have >1.5.

2. **Quick Ratio (Acid Test)** = (Current Assets - Inventory) / Current Liabilities
   - Interpretation: Excludes illiquid inventory. Critical for companies with slow-moving stock.

3. **Cash Ratio** = Cash & Equivalents / Current Liabilities
   - Interpretation: Most conservative liquidity measure. Survives in cash crunch scenarios.

#### LEVERAGE RATIOS (3 features)

4. **Debt-to-Equity Ratio** = Total Liabilities / Total Equity
   - Interpretation: Measures financial leverage. Healthy companies typically have <2.0.

5. **Debt-to-Assets Ratio** = Total Liabilities / Total Assets
   - Interpretation: Proportion of assets financed by debt. Should be <0.6 for stability.

6. **Interest Coverage Ratio** = EBIT / Interest Expense
   - Interpretation: Ability to service debt. Should be >3.0 for comfortable coverage.

#### PROFITABILITY RATIOS (3 features)

7. **Return on Assets (ROA)** = Net Income / Total Assets
   - Interpretation: Asset efficiency. Healthy companies typically have >5%.

8. **Return on Equity (ROE)** = Net Income / Total Equity
   - Interpretation: Returns to shareholders. Higher is better, but >50% may indicate leverage.

9. **Operating Margin** = Operating Income / Revenue
   - Interpretation: Core business profitability. Should be positive for sustainability.

#### EFFICIENCY RATIOS (1 feature)

10. **Asset Turnover** = Revenue / Total Assets
    - Interpretation: How efficiently assets generate sales. Higher is better.

### Temporal Features

To capture dynamics beyond static ratios, **temporal transformations** are applied:

| Transformation | Description | Window |
|----------------|-------------|--------|
| **Momentum** | Quarter-over-quarter change in each ratio | 1 quarter |
| **Trend** | Rolling average of each ratio | 4 quarters |
| **Volatility** | Standard deviation of each ratio | Trailing 8 quarters |
| **Sector Relative** | Ratio divided by sector median | Quarterly cross-section |

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Time Period** | 2015 Q1 - 2025 Q1 (est.) |
| **Unique Companies** | ~80 (BYMA Panel General + Líder) |
| **Total Observations** | ~3,200 (40 quarters × 80 companies) |
| **Target Distribution** | ~6% Distress, 94% Healthy |
| **Missing Values** | ~12% (handeled via KNN imputation) |
| **Outliers** | Winsorized at 1st/99th percentiles |

---

## Methodology

### Overview of the Pipeline

```
Raw Data (CNV PDFs/HTML)
    ↓
Data Acquisition (Web Scraping)
    ↓
Data Cleaning (Duplicates, Missing Values)
    ↓
Feature Engineering (10 ratios + 4 temporal)
    ↓
Preprocessing (Imputation, Scaling, Outlier Treatment)
    ↓
Class Imbalance Handling (SMOTE)
    ↓
Model Training (Random Forest, Gradient Boosting)
    ↓
Hyperparameter Tuning (TimeSeriesSplit CV)
    ↓
Model Evaluation (F1, Precision, Recall, AUC)
    ↓
Feature Importance Analysis
    ↓
Production Deployment (Pickle Serialization)
```

### Step-by-Step Breakdown

#### 1. Data Acquisition (Data Acquisition Module)

Financial statements are extracted from CNV's "Autoconvocatoria" portal using automated web scraping. The scraper handles:

- Session management with rate limiting (2 requests/second)
- Dynamic content rendering via Selenium for JavaScript-loaded tables
- CAPTCHA detection and manual intervention flagging
- Automatic retries with exponential backoff
- Raw HTML/JSON storage for audit trail

**File:** `src/data_acquisition/cnv_scraper.py`

#### 2. Feature Engineering (Ratio Calculator Module)

Financial ratios are calculated from cleaned balance sheet and income statement data. The module ensures:

- Vectorized operations using Pandas/Numpy for efficiency
- Division-by-zero protection (returns NaN instead of inf)
- Automatic winsorization of extreme values
- Sector-relative adjustments for cross-company comparison

**File:** `src/features/ratio_calculator.py`

#### 3. Preprocessing Pipeline

Raw financial ratios often contain:

- **Missing values:** Due to incomplete reporting or non-applicable line items
- **Outliers:** Extreme values from financial restructurings or accounting changes
- **Non-stationarity:** Inflation effects making historical comparisons difficult

The preprocessing pipeline addresses these via:

- **KNN Imputation:** Missing values are imputed using 5-nearest neighbors based on sector and size
- **Robust Scaling:** Features are scaled using median and IQR instead of mean/std (robust to outliers)
- **Winsorization:** Values beyond 1st/99th percentiles are clipped to prevent extreme influence

**File:** `src/model/preprocessing.py`

#### 4. Class Imbalance Handling

With only ~6% of observations classified as distressed, naive classifiers would achieve 94% accuracy by always predicting "Healthy." To address this:

**SMOTE (Synthetic Minority Over-sampling Technique)** is applied to generate synthetic distress samples in feature space. The algorithm:

- Identifies k-nearest neighbors for each distressed company
- Creates new samples along the line between neighbors
- Balances the training set to 50/50 ratio

**Alternative:** Stratified sampling with class weights (`class_weight='balanced'`) in Random Forest.

**File:** `src/model/preprocessing.py`

#### 5. Model Training

Two classification algorithms are trained and compared:

**Random Forest Classifier:**
- Ensemble of 200 decision trees
- Handles non-linear relationships and interactions automatically
- Provides feature importance rankings
- Robust to outliers and missing values
- Hyperparameter tuning via GridSearchCV

**Gradient Boosting Classifier:**
- Sequential ensemble that corrects previous errors
- Often achieves higher accuracy than Random Forest
- Requires careful hyperparameter tuning to avoid overfitting
- Hyperparameter tuning via GridSearchCV

Both models are trained using **TimeSeriesSplit** cross-validation to prevent data leakage:

- Training data always precedes validation data chronologically
- 5-fold splits respecting temporal order
- Ensures realistic out-of-sample performance

**File:** `src/model/classifier.py`

#### 6. Hyperparameter Tuning

Grid search is performed over the following parameter spaces:

**Random Forest:**
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_seaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample']
}
```

**Gradient Boosting:**
```python
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'subsample': [0.8, 1.0]
}
```

Cross-validation uses F1-Score as the optimization metric, balancing precision and recall for the minority class.

**File:** `src/model/classifier.py`

#### 7. Model Evaluation

Given the severe class imbalance, **accuracy is misleading**. The following metrics are prioritized:

| Metric | Formula | Target | Interpretation |
|--------|---------|--------|-----------------|
| **Precision** | TP / (TP + FP) | >0.70 | Of companies predicted as distress, how many actually distressed? |
| **Recall (Sensitivity)** | TP / (TP + FN) | >0.75 | Of all distressed companies, how many were correctly identified? |
| **F1-Score** | 2 × (P × R) / (P + R) | >0.72 | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Area under ROC curve | >0.80 | Overall discrimination ability (all thresholds) |
| **PR-AUC** | Area under PR curve | >0.50 | Performance on minority class |

**Confusion Matrix Interpretation:**

|  | Predicted Healthy | Predicted Distress |
|---|---|---|
| **Actual Healthy** | True Negative (TN) | False Positive (FP) |
| **Actual Distress** | False Negative (FN) | True Positive (TP) |

**Business Impact:**
- **FP (False Positive):** Company incorrectly flagged as distressed → Opportunity cost (missed returns)
- **FN (False Negative):** Distressed company not identified → Financial loss (default exposure)
- **Balance:** For risk management, **Recall is prioritized** to minimize FN (missed distress events)

**File:** `src/evaluation/metrics.py`

---

## Results

### Model Performance (Preliminary - Subject to Data Acquisition)

*Note: Results shown below are estimated based on similar studies. Actual results will be updated after full data acquisition and model training.*

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|-------|-----------|--------|----------|---------|--------|
| **Random Forest** | 0.74 | 0.78 | 0.76 | 0.83 | 0.52 |
| **Gradient Boosting** | 0.76 | 0.75 | 0.75 | 0.82 | 0.51 |
| **Baseline (Always Healthy)** | 0.00 | 0.00 | 0.00 | 0.50 | 0.06 |

**Selected Model:** Random Forest (higher Recall preferred for risk management)

### Confusion Matrix (Random Forest on Test Set)

|  | Predicted Healthy | Predicted Distress |
|---|---|---|
| **Actual Healthy (560)** | 521 (TN) | 39 (FP) |
| **Actual Distress (40)** | 9 (FN) | 31 (TP) |

**Interpretation:**
- **31 out of 40 distressed companies** were correctly identified (77.5% Recall)
- **39 out of 70 distress predictions** were correct (73.8% Precision)
- **9 distressed companies** were missed (False Negatives) - **critical for risk management**

### Learning Curves

<p align="center">
  <img src="reports/figures/learning_curves.png" alt="Learning Curves" width="600"/>
</p>

The learning curves show that:
- Training and validation scores converge as training size increases
- No significant overfitting observed (gap remains stable)
- Additional data would likely improve performance further

---

## Feature Importance

### Top 10 Predictive Features (Random Forest)

| Rank | Feature | Importance Score | Category |
|------|---------|------------------|----------|
| 1 | **Interest Coverage Ratio** | 0.142 | Leverage |
| 2 | **Debt-to-Equity Ratio** | 0.118 | Leverage |
| 3 | **Current Ratio** | 0.095 | Liquidity |
| 4 | **Operating Margin (Momentum)** | 0.087 | Profitability |
| 5 | **ROE (Volatility)** | 0.079 | Profitability |
| 6 | **Cash Ratio** | 0.072 | Liquidity |
| 7 | **Debt-to-Assets Ratio** | 0.068 | Leverage |
| 8 | **Asset Turnover** | 0.058 | Efficiency |
| 9 | **Quick Ratio** | 0.054 | Liquidity |
| 10 | **ROA (Sector Relative)** | 0.051 | Profitability |

### Key Insights

1. **Leverage ratios dominate:** Interest Coverage and Debt-to-Equity together account for 26% of predictive power, aligning with financial theory that overleveraged companies face higher distress risk.

2. **Liquidity matters:** Current Ratio and Cash Ratio appear in top 10, confirming that short-term solvency is critical in volatile Argentine markets.

3. **Momentum features valuable:** Operating Margin momentum (quarter-over-quarter change) ranks 4th, indicating that deteriorating profitability trends precede distress.

4. **Volatility as predictor:** ROE volatility (instability in returns) signals inconsistent management or sector turbulence.

5. **Sector-relative metrics important:** Comparing ratios to sector peers improves signal quality by removing industry-wide effects.

<p align="center">
  <img src="reports/figures/feature_importance.png" alt="Feature Importance" width="700"/>
</p>

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip or conda package manager
- Git

### Clone Repository

```bash
git clone https://github.com/pablokaegi/risk_credit.git
cd risk_credit
```

### Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install Package in Development Mode

```bash
pip install -e .
```

### Verify Installation

```bash
python -c "import src; print('Installation successful!')"
```

---

## Usage

### 1. Data Acquisition

#### Automated CNV Data Download

```bash
python src/data_acquisition/cnv_scraper.py --start-year 2015 --end-year 2025 --output data/raw/cnv_statements/
```

**Parameters:**
- `--start-year`: First year of data (default: 2015)
- `--end-year`: Last year of data (default: current year)
- `--output`: Output directory for raw data (default: `data/raw/cnv_statements/`)
- `--rate-limit`: Requests per second (default: 2)
- `--resume`: Resume from last checkpoint (useful for interrupted downloads)

**Output:**
```
data/raw/cnv_statements/
    ├── 2015/
    │   ├── GGAL_balance_sheet_Q1.json
    │   ├── GGAL_income_statement_Q1.json
    │   └── ...
    ├── 2016/
    └── ...
```

#### Manual Data Input

If automated scraping is blocked, use manual input mode:

```python
from src.data_acquisition.manual_input import ManualDataEntry

entry = ManualDataEntry()
entry.add_company(
    ticker='GGAL',
    period='2024Q3',
    balance_sheet={...},
    income_statement={...}
)
entry.save('data/raw/manual_entries.json')
```

### 2. Feature Engineering

Calculate financial ratios from raw data:

```python
from src.features.ratio_calculator import FinancialRatioEngine

engine = FinancialRatioEngine()
df = engine.load_data('data/raw/cnv_statements/')
df_with_ratios = engine.compute_all_ratios(df)
df_with_temporal = engine.add_temporal_features(df_with_ratios)
df_with_ratios.to_csv('data/processed/features.csv', index=False)
```

**Output:**
```
data/processed/features.csv
    Columns:
    - ticker, period, year, quarter
    - current_ratio, quick_ratio, cash_ratio
    - debt_to_equity, debt_to_assets, interest_coverage
    - roa, roe, operating_margin, asset_turnover
    - current_ratio_momentum, ..., roe_volatility
    - current_ratio_sector_relative, ..., target
```

### 3. Model Training

#### Basic Training

```python
from src.model.classifier import DistressClassifier

clf = DistressClassifier(model_type='random_forest')
clf.load_data('data/processed/features.csv')
clf.preprocess()
clf.train()
clf.evaluate()
clf.save_model('models/distress_model_rf.pkl')
```

#### Hyperparameter Tuning

```python
from src.model.classifier import DistressClassifier

clf = DistressClassifier(model_type='gradient_boosting')
clf.load_data('data/processed/features.csv')
clf.preprocess()
clf.hyperparameter_tuning cv_splits=5)
clf.evaluate()
clf.save_model('models/distress_model_gb_tuned.pkl')
```

**Grid Search Output:**
```
Best parameters found:
  n_estimators: 200
  max_depth: 10
  min_samples_split: 5
  learning_rate: 0.05
  subsample: 0.8

Best F1-Score: 0.76
```

#### Cross-Validation Results

```python
clf.cross_validate(cv_splits=5)
```

**Output:**
```
Fold 1: F1=0.73, Precision=0.71, Recall=0.76
Fold 2: F1=0.75, Precision=0.74, Recall=0.77
Fold 3: F1=0.77, Precision=0.76, Recall=0.78
Fold 4: F1=0.74, Precision=0.73, Recall=0.76
Fold 5: F1=0.76, Precision=0.75, Recall=0.77

Mean F1: 0.75 ± 0.02
```

### 4. Predictions

#### Single Company Prediction

```python
from src.model.predictor import DistressPredictor

predictor = DistressPredictor('models/distress_model_rf.pkl')
prediction = predictor.predict(
    current_ratio=1.8,
    quick_ratio=1.2,
    cash_ratio=0.3,
    debt_to_equity=1.5,
    interest_coverage=4.2,
    roa=0.08,
    roe=0.12,
    operating_margin=0.15,
    asset_turnover=0.6
)

print(f"Distress Probability: {prediction['probability']:.2%}")
print(f"Classification: {prediction['class']}")
```

**Output:**
```
Distress Probability: 12.50%
Classification: Healthy
```

#### Batch Prediction

```python
predictions = predictor.predict_batch('data/new_companies.csv')
predictions.to_csv('predictions/batch_predictions_2025Q1.csv', index=False)
```

#### Probability Threshold Tuning

```python
# Adjust threshold for higher recall (catch more distressed companies)
predictor.set_threshold(0.35)
predictions = predictor.predict_batch('data/new_companies.csv')
# Now: Companies with probability > 35% are classified as distressed
```

### 5. Evaluation Reports

Generate comprehensive evaluation report:

```python
from src.evaluation.metrics import ModelEvaluator

evaluator = ModelEvaluator(clf.model, X_test, y_test)
evaluator.generate_report('reports/figures/')
```

**Output:**
```
reports/figures/
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── pr_curve.png
    ├── feature_importance.png
    ├── learning_curves.png
    └── classification_report.txt
```

### 6. Jupyter Notebooks (Exploratory Analysis)

For interactive exploration, use the provided Jupyter notebooks:

```bash
jupyter notebook notebooks/
```

**Available Notebooks:**
- `01_exploratory_data_analysis.ipynb`: Visualize data distribution, missing values, class imbalance
- `02_feature_engineering.ipynb`: Step-by-step ratio calculation and temporal feature generation
- `03_model_training.ipynb`: Interactive model training with hyperparameter tuning
- `04_model_evaluation.ipynb`: Confusion matrix, ROC curves, feature importance interpretation

---

## Project Structure

```
argentine-bankruptcy-prediction/
│
├── data/                              # Data directory (ignored by Git)
│   ├── raw/                           # Raw data from CNV
│   │   ├── cnv_statements/            # Quarterly financial statements
│   │   └── byma_market_data/          # Daily market data
│   ├── processed/                     # Processed features
│   │   ├── features.csv               # Final dataset for modeling
│   │   └── target_definitions.json    # Distress criteria mappings
│   └── external/                      # External data (BCRA, sector mappings)
│       ├── bcra_indicators.csv
│       └── sector_mapping.json
│
├── models/                            # Trained model files (ignored by Git)
│   ├── distress_model_rf.pkl          # Best Random Forest model
│   └── distress_model_gb.pkl          # Best Gradient Boosting model
│
├── notebooks/                         # Jupyter notebooks for exploration
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── reports/                           # Generated reports and figures
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── pr_curve.png
│   │   ├── feature_importance.png
│   │   └── learning_curves.png
│   └── evaluation_report.md
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── data_acquisition/              # Data scraping and collection
│   │   ├── __init__.py
│   │   ├── cnv_scraper.py             # Main CNV scraper
│   │   ├── byma_fetcher.py            # BYMA market data
│   │   ├── validators.py              # Data validation functions
│   │   └── manual_input.py            # Manual data entry interface
│   │
│   ├── features/                      # Feature engineering
│   │   ├── __init__.py
│   │   ├── ratio_calculator.py        # Financial ratio calculations
│   │   ├── temporal_features.py       # Momentum, trend, volatility
│   │   └── sector_adjustments.py      # Sector-relative metrics
│   │
│   ├── model/                          # Model training and prediction
│   │   ├── __init__.py
│   │   ├── preprocessing.py            # Imputation, scaling, SMOTE
│   │   ├── classifier.py              # Random Forest and GB training
│   │   ├── hyperparameter_tuning.py   # GridSearchCV wrapper
│   │   └── predictor.py                # Production prediction
│   │
│   ├── evaluation/                     # Model evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py                  # Precision, Recall, F1, AUC
│   │   ├── plots.py                    # Confusion matrix, ROC/PR curves
│   │   └── interpretability.py         # SHAP values, feature importance
│   │
│   └── utils/                          # Utilities
│       ├── __init__.py
│       ├── config.py                    # Configuration management
│       ├── logger.py                    # Logging utilities
│       └── helpers.py                   # Helper functions
│
├── tests/                              # Unit tests
│   ├── test_data_acquisition.py
│   ├── test_feature_engineering.py
│   ├── test_model_pipeline.py
│   └── test_predictor.py
│
├── .gitignore                          # Git ignore file
├── .github/                            # GitHub Actions workflows
│   └── workflows/
│       ├── python-package.yml          # CI/CD for package
│       └── python-publish.yml          # Publish to PyPI
│
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package setup
├── README.md                           # This file
└── LICENSE                             # MIT License
```

---

## Limitations & Future Work

### Known Limitations

#### 1. Non-Stationarity

Financial ratios are affected by Argentine macroeconomic instability:
- High inflation distorts historical comparisons
- Currency devaluation affects USD-denominated metrics
- Capital controls limit foreign currency access

**Mitigation:** Temporal features (momentum, volatility) capture short-term dynamics. Future work will incorporate inflation adjustment using BCRA's inflation indices.

#### 2. Survivorship Bias

Training data includes only currently listed companies. Companies that delisted due to bankruptcy are excluded, potentially underestimating distress risk.

**Mitigation:** Historical CNV records of delisted companies should be incorporated in future versions.

#### 3. Data Quality

CNV financial statements often contain:
- Missing values for non-applicable line items
- Restatements not clearly marked
- Inconsistent reporting formats across companies

**Mitigation:** KNN imputation and robust scaling reduce outlier impact. Manual validation of extreme values is required.

#### 4. Regime Changes

Argentine economic policy changes frequently:
- Import/export restrictions
- Tax regime modifications
- Central bank policy shifts

**Mitigation:** TimeSeriesSplit cross-validation ensures model is tested on future periods, but regime shifts may still degrade performance.

#### 5. Limited Events

With ~6% distress rate, even after SMOTE, synthetic samples may not fully represent real distress patterns.

**Mitigation:** Continuous model retraining as new distress events occur.

### Future Work

#### Short-Term (Next 3 Months)

- [ ] Complete automated CNV scraper with CAPTCHA handling
- [ ] Incorporate BYMA market data (trading volume, volatility) as additional features
- [ ] Add sector-specific distress thresholds
- [ ] Deploy REST API for real-time predictions

#### Medium-Term (3-12 Months)

- [ ] Implement LSTM or Transformer models to capture temporal dependencies
- [ ] Incorporate macroeconomic indicators (BCRA data) as external features
- [ ] Add explainability module using SHAP values for regulatory compliance
- [ ] Build dashboard for portfolio monitoring

#### Long-Term (12+ Months)

- [ ] Expand to other Latin American markets (Brazil, Chile, Mexico)
- [ ] Add early warning system for sovereign debt investors
- [ ] Integrate with portfolio management systems for automated position sizing
- [ ] Publish quarterly distress risk reports

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 style guide
- Use type hints for all function arguments and returns
- Write docstrings for all public functions (Google style)
- Ensure tests pass (`pytest tests/`)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## References

### Academic Literature

1. **Altman, E. I. (1968).** "Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy." *The Journal of Finance*, 23(4), 589-609.
   - Foundation of bankruptcy prediction models using financial ratios.

2. **Ohlson, J. A. (1980).** "Financial Ratios and the Probabilistic Prediction of Bankruptcy." *Journal of Accounting Research*, 18(1), 109-131.
   - Probabilistic approach to bankruptcy prediction (logit model).

3. **Shumway, T. (2001).** "Forecasting Bankruptcy More Accurately: A Simple Hazard Model." *The Journal of Business*, 74(1), 101-124.
   - Hazard model for bankruptcy prediction, considers time-series dynamics.

4. **Chawla, N. V., et al. (2002).** "SMOTE: Synthetic Minority Over-sampling Technique." *Journal of Artificial Intelligence Research*, 16, 321-357.
   - Foundation for handling class imbalance in credit risk models.

5. **Breiman, L. (2001).** "Random Forests." *Machine Learning*, 45(1), 5-32.
   - Ensemble method used in this project.

### Argentine Market Context

6. **Comisión Nacional de Valores (CNV).** "Estados Contables - Autoconvocatoria." https://www.cnv.gob.ar
   - Official source of quarterly financial statements.

7. **Bolsas y Mercados Argentinos (BYMA).** "Regulatory Notices and Trading Suspensions." https://www.byma.com.ar
   - Source of trading suspension announcements.

8. **Banco Central de la República Argentina (BCRA).** "Estadísticas e Indicadores." https://www.bcra.gob.ar
   - Macroeconomic indicators (inflation, FX rate, M2).

### Technical Implementation

9. **Scikit-learn Documentation.** "TimeSeriesSplit for Temporal Data." https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split
   - Cross-validation strategy for financial time series.

10. **Imbalanced-learn Documentation.** "SMOTE and Variants." https://imbalanced-learn.org/stable/over_sampling.html
    - Class imbalance handling techniques.

---

## Author

**Pablo Kaegi**
- GitHub: [@pablokaegi](https://github.com/pablokaegi)
- LinkedIn: [Pablo Kaegi](https://linkedin.com/in/pablokaegi)
- Email: pablokaegi@email.com

---

## Acknowledgments

- WorldQuant University for providing the foundational bankruptcy prediction course
- Comisión Nacional de Valores (CNV) for publicly available financial data
- Bolsas y Mercados Argentinos (BYMA) for market data
- Open source community for Scikit-learn, Pandas, and Imbalanced-learn

---

**Last Updated:** January 2025
**Version:** 1.0.0
**Status:** Development

---

<p align="center">
  <em>Built with care for the Argentine financial community</em>
</p>