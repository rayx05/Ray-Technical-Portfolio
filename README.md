# Project 1:Depression Prediction Using Machine Learning

Predicting depression diagnosis in students using survey data and supervised 
machine learning models.

## Problem Statement

Mental health screening at scale is challenging — this project explores whether 
student depression can be predicted from observable behavioral and academic 
factors using machine learning, without requiring clinical diagnosis.

## Dataset

- **Source:** Student Depression Dataset (Kaggle)
- **Size:** 27,901 records, filtered to 27,837 student observations
- **Features:** Academic pressure, CGPA, sleep duration, dietary habits, 
financial stress, suicidal thoughts history, family mental health history, 
and more
- **Target:** Binary depression diagnosis (0 = No, 1 = Yes)

## Methods

**Data Preprocessing**
- Filtered dataset to student population only
- Converted categorical variables to numeric 
  (sleep duration, dietary habits, gender, suicidal thoughts)
- Dropped high-cardinality features (city, degree) to reduce noise
- Handled missing values via row removal

**Models Evaluated**
- Decision Tree Classifier (depth-optimized via GridSearchCV)
- Naive Bayes
- Logistic Regression
- Linear SVC
- Linear Regression (for comparison)

**Validation**
- Train/test split (75/25)
- 5-fold cross-validation for all classifiers

## Key Results

| Model | Accuracy |
|---|---|
| Decision Tree (depth=7) | 82.8% |
| Cross-validated Decision Tree | 84.6% |
| Linear Regression (R²) | 0.517 |

**Top predictive features (Decision Tree):**
Suicidal thoughts history, academic pressure, and financial stress emerged 
as the strongest predictors of depression diagnosis.

## Tools

Python, pandas, scikit-learn, matplotlib, seaborn, Google Colab

## How to Run

Open `Predicting Depression Diagnosis with Survey Result.ipynb` in Jupyter or Google Colab.
Update the dataset path if running locally.

# Project 2: Birth Quarter and Returns to Education — Causal Inference Study

Estimating the causal effect of education on earnings using birth quarter 
as an instrumental variable, applied to a 329,509-record census dataset.

## Research Question

Does education causally increase earnings — and by how much? Standard OLS 
estimates are biased by ability and selection. This study uses quarter of 
birth as an instrument to isolate exogenous variation in education levels 
and produce unbiased causal estimates.

## Dataset

- **Source:** U.S. Census data
- **Size:** 329,509 individuals born 1930–1939
- **Variables:** Years of education, log wage, quarter of birth, year of 
birth, state of residence, marital status, metropolitan area status, 
race, region

## Methods

**Exploratory Analysis**
- Documented a consistent seasonal pattern: individuals born in Q1 
  receive less education on average than those born in Q4 of the 
  prior year — Q1 birth predicts lower education in 100% of cohorts
- Visualized quarter-over-quarter change in average education across 
  40 year-quarter cohorts

**OLS Regression**
- Regressed years of education on quarter of birth dummies, 
  controlling for centered year of birth
- Q1 birth associated with −0.148 fewer years of education 
  (p < 0.01) relative to Q4

**Two-Stage Least Squares (2SLS)**
- Stage 1: Instrumented years of education using quarter of birth, 
  year of birth fixed effects, and state fixed effects
- Stage 2: Estimated causal effect of instrumented education on 
  log wages, controlling for metropolitan area, marital status, 
  race, and region

## Key Results

| Model | Return to Education (log wage) |
|---|---|
| OLS (baseline) | 0.067*** |
| TSLS (baseline) | 0.104*** |
| OLS (full controls) | 0.063*** |
| TSLS (full controls) | 0.096*** |

TSLS estimates consistently exceed OLS estimates — consistent with 
downward ability bias in OLS. Each additional year of education 
increases earnings by approximately 9.6–10.4% after controlling 
for endogeneity.

## Tools

R, regression modeling, instrumental variables, fixed effects, 
data visualization

## How to View

Open the presentation PDF in this repository for full methodology, 
visualizations, and regression output tables.
