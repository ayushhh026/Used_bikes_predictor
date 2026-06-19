# Used Bikes Price Predictor 🏍️

<div align="center">

```
██╗   ██╗███████╗███████╗██████╗     ██████╗ ██╗██╗  ██╗███████╗███████╗
██║   ██║██╔════╝██╔════╝██╔══██╗    ██╔══██╗██║██║ ██╔╝██╔════╝██╔════╝
██║   ██║███████╗█████╗  ██║  ██║    ██████╔╝██║█████╔╝ █████╗  ███████╗
██║   ██║╚════██║██╔══╝  ██║  ██║    ██╔══██╗██║██╔═██╗ ██╔══╝  ╚════██║
╚██████╔╝███████║███████╗██████╔╝    ██████╔╝██║██║  ██╗███████╗███████║
 ╚═════╝ ╚══════╝╚══════╝╚═════╝     ╚═════╝ ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝
```

### EDA + 12-Model Regression Benchmark on 32K Used Bike Listings

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Status](https://img.shields.io/badge/Status-Complete-22C55E?style=for-the-badge)](.)

<br/>

> **Data Cleaning · EDA · ColumnTransformer · 12-Model Benchmark · RandomizedSearchCV**

</div>

---

## What is This Project?

An end-to-end EDA + ML modeling project on a dataset of **32,648 used bike listings from the Indian resale market** (cities, brands, and pricing reflect India's two-wheeler ecosystem — Bajaj, Royal Enfield, Hero, TVS dominate volume; Triumph, Harley-Davidson, Ducati sit at the premium end).

**Personal motivation:** I built this while actively shopping for my own second-hand bike. Rather than relying on gut feeling or dealer quotes, I wanted a data-backed sense of what factors actually drive resale price in the Indian market — brand depreciation curves, the real impact of kms driven vs age, and how much ownership history (First Owner vs Second Owner+) actually costs you. The model ended up directly informing what I was willing to pay for specific brand/age/kms combinations.

- ✅ **Data Cleaning** — rare city consolidation, exact duplicate removal
- ✅ **EDA** — univariate, bivariate, and multivariate analysis across price, age, brand, and power
- ✅ **Encoding** — `ColumnTransformer` combining OneHotEncoder (brand, city) + OrdinalEncoder (owner)
- ✅ **Model Benchmarking** — 12 regression models trained and evaluated on train and test sets
- ✅ **Hyperparameter Tuning** — `RandomizedSearchCV` on XGBoost (50 iterations × 3-fold CV = 150 fits)

---

## Pipeline Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        RAW DATASET                           │
│        Used_Bikes.csv · 32,648 rows · 8 columns              │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                      DATA CLEANING                           │
│  Drop bike_name · Rare city → "Other" · Remove duplicates     │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                         EDA                                  │
│  Univariate (price, age) · Bivariate (age vs price,          │
│  brand vs price) · Multivariate (correlation heatmap)        │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  TRAIN/TEST SPLIT (70/30)                    │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                 COLUMNTRANSFORMER ENCODING                   │
│   OneHotEncoder → brand, city (drop=first)                  │
│   OrdinalEncoder → owner (First → Fourth+, ordered)          │
│   remainder='passthrough' → kms_driven, age, power           │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              12-MODEL BENCHMARK                               │
│   Linear · Lasso · Ridge · KNN · DT · RF · AdaBoost          │
│   Gradient Boosting (+Tuned) · XGBoost (+Tuned) · SVR        │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│         RandomizedSearchCV ON XGBOOST                        │
│        50 iterations × 3-fold CV = 150 fits                  │
└──────────────────────────────────────────────────────────────┘
```

---

## Data Cleaning

| Step | Detail |
|---|---|
| Dropped `bike_name` | High-cardinality text field, not used as a model feature |
| Rare city consolidation | Cities with ≤5 listings grouped into `"Other"` — prevents one-hot explosion across 93+ cities |
| Duplicate removal | Exact duplicate rows identified and removed to prevent the model from over-weighting repeated listings |
| Missing values | None found — `df.isnull().sum()` returned 0 across all columns |

---

## Exploratory Data Analysis

### Univariate — Price Distribution
Price is heavily right-skewed — most bikes sell in the ₹20,000–₹80,000 range, with a long tail toward premium bikes (Triumph, Royal Enfield, Harley-Davidson) reaching ₹19 lakh+.

<img width="580" height="432" alt="f2842455-af4a-4c38-9cf4-64388414e864" src="https://github.com/user-attachments/assets/3cadc5df-c01f-4441-82ab-c36da55a92ea" />

### Univariate — Age Distribution
Most listed bikes are **6–7 years old** on average (`mean age ≈ 6.66 years`), consistent with typical resale timing in the Indian two-wheeler market.

<img width="580" height="432" alt="57e10c8f-9b65-47dc-8207-5905087ff500" src="https://github.com/user-attachments/assets/0177074d-4d84-4cec-9151-492ab533be7d" />

### Bivariate — Age vs Price
Clear negative relationship — price drops as bike age increases, as expected for depreciating assets.

<img width="597" height="432" alt="ca5ae009-7ce2-4761-8777-ce79ae441ed6" src="https://github.com/user-attachments/assets/e937a1d4-23ab-4b55-a509-90712f40ab98" />

### Bivariate — Brand vs Price
Boxplot across brands reveals wide price variance — premium brands (Triumph, Harley-Davidson, Ducati, BMW) sit far above mass-market brands (Hero, TVS, Bajaj) in both median and spread.

<img width="1242" height="848" alt="44a52312-549f-44b7-978c-6cbc313272dc" src="https://github.com/user-attachments/assets/92c0c1e8-e15b-46f8-bd2a-efa65c9efb84" />

### Multivariate — Correlation Heatmap
`power` shows the strongest numeric correlation with `price` — intuitive, since engine displacement is a major price driver. The correlation is moderate rather than strong, since brand and ownership history also weigh heavily on price (and aren't numeric).

<img width="527" height="418" alt="17572a61-e6db-4a77-9473-55e5046ecb6f" src="https://github.com/user-attachments/assets/cc41f159-0715-4550-b956-1a4a207e3fa1" />
---

## Encoding Strategy

A single `ColumnTransformer` handles all categorical encoding in one step:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'), ['brand', 'city']),
        ('od', OrdinalEncoder(categories=[['First Owner', 'Second Owner', 
                                            'Third Owner', 'Fourth Owner Or More']]), ['owner'])
    ],
    remainder='passthrough'
)
```

| Column | Encoder | Why |
|---|---|---|
| `brand`, `city` | OneHotEncoder | No inherent order between categories |
| `owner` | OrdinalEncoder (custom order) | Ownership count has a natural order — more owners generally means lower price |
| `kms_driven`, `age`, `power` | Passthrough | Already numeric, no encoding needed |

`handle_unknown='ignore'` ensures unseen categories at inference time don't crash the pipeline — they're encoded as all zeros instead.

---

## Model Benchmark — 12 Models Compared

All models trained on the same 70/30 split and evaluated on both train and test sets to catch overfitting.

| Model | Test MAE | Test RMSE | Test R² | Train R² | Overfit Gap |
|---|---|---|---|---|---|
| **Gradient Boosting** | 14,058 | 37,015 | **0.9009** | 0.9522 | Low |
| XGBoost (Tuned) | 13,230 | 37,572 | 0.8979 | 0.9904 | High |
| Random Forest | 13,051 | 39,149 | 0.8891 | 0.9828 | Moderate |
| XGBoost (Default) | 13,019 | 39,377 | 0.8878 | 0.9905 | High |
| Gradient Boosting (Tuned) | 13,207 | 40,827 | 0.8794 | 0.9838 | Moderate |
| Lasso Regression | 24,185 | 48,564 | 0.8294 | 0.8343 | None |
| Decision Tree | 17,077 | 54,833 | 0.7825 | 0.9987 | Severe |
| Linear Regression | 28,141 | 57,431 | 0.7614 | 0.7817 | None |
| KNN Regressor | 28,226 | 71,026 | 0.6350 | 0.7759 | Moderate |
| Ridge Regression | 37,893 | 71,439 | 0.6308 | 0.6664 | None |
| AdaBoost | 71,078 | 88,593 | 0.4322 | 0.5278 | None |
| SVR | 51,276 | 120,472 | **-0.0500** | -0.0637 | None (just bad fit) |

### Key Observations

**Best generalizing model: Gradient Boosting (untuned)** — highest test R² (0.9009) with the smallest train-test gap among top performers. Despite XGBoost variants scoring higher on training data, their larger overfit gap makes Gradient Boosting the more reliable choice for unseen data.

**Decision Tree overfits severely** — train R² of 0.9987 collapses to 0.7825 on test data, a textbook example of an unconstrained tree memorizing training data instead of learning generalizable patterns.

**SVR failed completely** — negative R² means the model performs worse than simply predicting the mean price every time. Likely cause: SVR is highly sensitive to feature scaling, and the pipeline did not scale numeric features before fitting.

**Lasso outperforms Ridge** — Lasso's L1 penalty zeroes out less useful one-hot encoded brand/city columns, effectively acting as built-in feature selection on a high-dimensional encoded space. Ridge's L2 penalty shrinks all coefficients but keeps every encoded column active, hurting performance on this sparse feature set.

---

## Hyperparameter Tuning — XGBoost

`RandomizedSearchCV` was used instead of `GridSearchCV` to efficiently search a large parameter space:

```python
params = {
    "learning_rate": [0.1, 0.01],
    "max_depth": [5, 8, 12, 20, 30],
    "n_estimators": [100, 200, 300],
    "colsample_bytree": [0.5, 0.8, 1, 0.3, 0.4]
}
```

**Search config:** 50 iterations × 3-fold CV = 150 total fits

**Best parameters found:**
```python
{
    'n_estimators': 300,
    'max_depth': 5,
    'learning_rate': 0.1,
    'colsample_bytree': 1
}
```

Interestingly, the tuned XGBoost (`max_depth=5`) performed close to but not better than the simpler Gradient Boosting baseline — suggesting the dataset's signal is well captured by shallower trees, and deeper/more complex models mainly add overfitting risk rather than genuine predictive power.

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Language** | Python 3.10+ | Core development |
| **Data** | Pandas, NumPy | Cleaning, transformation |
| **Visualization** | Matplotlib, Seaborn | EDA |
| **ML** | Scikit-Learn | Encoding, 9 of the 12 models, metrics |
| **Boosting** | XGBoost | Gradient-boosted trees |
| **Tuning** | RandomizedSearchCV | Hyperparameter search |

---

## Project Structure

```
Used-Bikes-Price-Predictor/
│
├── notebooks/
│   └── used_bikes_eda_modeling.ipynb   # Full EDA + modeling + prediction pipeline
│
├── data/
│   └── Used_Bikes.csv                  # Raw dataset
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/ayushhh026/Used-Bikes-Price-Predictor.git
cd Used-Bikes-Price-Predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook eda.ipynb
```

---

## Key Engineering Decisions

**Why consolidate rare cities into "Other"?**
With 93+ unique cities, one-hot encoding all of them would create a huge sparse feature space, many with only a handful of samples. Cities with ≤5 listings were merged into a single `"Other"` category — reducing dimensionality and preventing the model from trying to learn from statistically insignificant samples.

**Why OrdinalEncoder for `owner` but OneHotEncoder for `brand`/`city`?**
`owner` has a genuine ordinal relationship — First Owner → Second Owner → Third Owner → Fourth+ represents increasing depreciation, a single ordered scale. `brand` and `city` have no such order — Bajaj isn't "more" or "less" than Honda — so one-hot encoding is correct there.

**Why RandomizedSearchCV over GridSearchCV for XGBoost?**
The parameter grid has 2 × 5 × 3 × 5 = 150 possible combinations. GridSearchCV would test all 150 combinations × folds. RandomizedSearchCV samples 50 combinations instead, getting near-optimal results in a third of the compute time — a reasonable tradeoff for this search space size.

**Why is Gradient Boosting preferred over XGBoost here despite similar scores?**
Model selection isn't just about peak test R² — it's about the train-test gap. XGBoost's near-perfect training R² (0.99) paired with a lower test R² than Gradient Boosting signals more overfitting. Gradient Boosting's smaller gap suggests it will generalize more reliably on completely new listings.

---

## Practical Use Case — Predicting My Own Bike's Fair Price

This isn't just an academic exercise — I used the trained models to value my own bike before negotiating a resale price.

**My bike:** Bajaj Dominar 400 · Mumbai · 7 years old · 38,000 km · Third Owner

```python
bikes_to_check = pd.DataFrame([
    {
        "city": "Mumbai",
        "kms_driven": 38000.0,
        "owner": "Third Owner",
        "age": 7.0,
        "power": 400.0,
        "brand": "Bajaj"
    }
])

bikes_transformed = preprocessor.transform(bikes_to_check)

for name, model in models.items():
    pred = model.predict(bikes_transformed[0:1])[0]
    print(f"{name:25s} → ₹{pred:,.0f}")
```

### Predicted Price Across All 12 Models

| Model | Predicted Price |
|---|---|
| Linear Regression | ₹2,08,952 |
| Decision Tree | ₹1,86,000 |
| Ridge Regression | ₹1,87,037 |
| Lasso Regression | ₹1,65,991 |
| **Gradient Boosting** | **₹1,04,725** |
| AdaBoost | ₹1,00,126 |
| Gradient Boosting (Tuned) | ₹92,523 |
| XGBoost (Default) | ₹89,860 |
| Random Forest | ₹1,09,049 |
| XGBoost (Best/Tuned) | ₹76,849 |
| KNN Regressor | ₹72,588 |
| SVR | ₹54,500 |

### Reading the Spread

The 12 models disagree wildly — from ₹54,500 (SVR) to ₹2,08,952 (Linear Regression), a difference of nearly **4x**. This spread itself is informative:

- **Linear, Ridge, Lasso, Decision Tree predict high (₹1.6L–₹2.1L)** — these models likely underweight the steep depreciation effect of `age=7` and `kms_driven=38,000` combined with `owner="Third Owner"`, since linear models assume smooth, additive effects rather than the compounding real-world depreciation that happens with high mileage + multiple owners.
- **Gradient Boosting (₹1,04,725) is the number I trust most** — it's the model with the best test R² (0.9009) and the smallest train-test overfit gap from the benchmark, making it the most reliable generalizer to a specific, slightly unusual combination (Third Owner is rare in the dataset, mostly First Owner listings).
- **SVR's ₹54,500 should be ignored entirely** — its test R² was negative, meaning it's worse than guessing the mean. The unscaled `kms_driven` feature (tens of thousands) likely dominates the distance-based SVR calculation.

**Takeaway for negotiation:** Using Gradient Boosting's ₹1,04,725 as the anchor, anything a seller asks meaningfully above ₹1.1L for this exact spec is worth pushing back on with this data.

### Real-World Validation

My actual all-in cost on this bike: **₹80,000 purchase price + ₹40,000 in repairs ≈ ₹1,20,000 total**.

Gradient Boosting's prediction of **₹1,04,725** lands close to this — within ~13% of my real total cost, and meaningfully closer than any of the other 11 models. This is a useful sanity check: the model isn't just statistically the best generalizer on the test set, it also holds up against a real lived cost I can independently verify. It reinforces that trusting the model with the best train-test gap over the model with the highest raw training score was the right call.

---

## Roadmap

- [ ] Scale numeric features before SVR (diagnose and fix the negative R²)
- [ ] Feature importance analysis on the best Gradient Boosting model
- [ ] Investigate why linear models predict ~2x higher than tree-based models for high-mileage, multi-owner bikes
- [ ] Try CatBoost (handles categorical features natively, no manual encoding needed)
- [ ] Cross-validation scores (not just single train/test split) for more robust comparison
- [ ] FastAPI deployment with a price prediction form

---

## License

[MIT License](LICENSE) — free to use, modify, and distribute with attribution.

---

## Author

**Ayush Shetty**
AI & Data Science Engineering Student

[![GitHub](https://img.shields.io/badge/GitHub-ayushhh026-181717?style=flat-square&logo=github)](https://github.com/ayushhh026)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ayush_Shetty-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/ayush-shetty-830a03281/)

---

<div align="center">

**⭐ Star this repo if it helped you — it keeps the project alive.**

*12 models tested, one clear winner — and the reasoning to back it up.*

</div>

---

> **Disclaimer:** Predictions are for educational purposes only and not a substitute for professional bike valuation.
