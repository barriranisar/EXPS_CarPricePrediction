# 🚗 Car Price Prediction with Machine Learning (EXPS Nexus Task 3)

An end-to-end Machine Learning regression project to accurately predict vehicle market valuations based on vehicle specifications, engine attributes, dimensions, fuel economy, and brand goodwill.

---

## 📌 Project Overview
- **Objective:** Build a robust regression pipeline to estimate car prices and identify the key physical and mechanical drivers influencing car valuation.
- **Dataset:** `Car_features_dataset.csv` (205 records, 26 raw features).
- **Target Variable:** `price` (USD).
- **Key Techniques:** Feature Engineering, Categorical Encoding, Feature Standardization, Multi-Model Benchmarking, Feature Importance Analysis, and Ensemble Consensus Inference.

---

## ⚙️ Machine Learning Pipeline Workflow

### 1. Data Cleaning & Feature Engineering
- **Brand Extraction:** Extracted make/brand names from `CarName` and corrected spelling typos (e.g., `toyouta` $\rightarrow$ `toyota`, `maxda` $\rightarrow$ `mazda`, `vokswagen`/`vw` $\rightarrow$ `volkswagen`, `porcshce` $\rightarrow$ `porsche`)[cite: 1].
- **Numeric Word Mapping:** Transformed text numbers (`doornumber`, `cylindernumber`) into actual integer quantities (e.g., `'four'` $\rightarrow$ `4`, `'six'` $\rightarrow$ `6`)[cite: 1].
- **Dimensionality Reduction:** Dropped uninformative identifier columns (`car_ID`, `CarName`)[cite: 1].

### 2. Categorical Encoding & Feature Scaling
- Applied **One-Hot Encoding** (`pd.get_dummies(..., drop_first=True)`) to categorical features (`Brand`, `fueltype`, `aspiration`, `carbody`, `drivewheel`, `enginelocation`, `enginetype`, `fuelsystem`) to avoid multicollinearity[cite: 1].
- Scaled all continuous and numeric features using **`StandardScaler`** ($z = \frac{x - \mu}{\sigma}$) on an 80/20 Train-Test split[cite: 1].

### 3. Model Training & Evaluation
Trained and evaluated state-of-the-art tree-based ensemble regression algorithms[cite: 1]:
- **Random Forest Regressor** (`n_estimators=100`)[cite: 1]
- **Gradient Boosting Regressor**[cite: 1]

---

## 📊 Model Performance Comparison

| Model | $R^2$ Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
| :--- | :---: | :---: | :---: |
| **Random Forest Regressor** | **0.9586** | **$1,288.69** | **$1,808.72** |
| **Gradient Boosting Regressor** | 0.9272 | $1,680.31 | $2,397.17 |

> Both models achieved exceptional accuracy, with Random Forest explaining **~95.86%** of total price variance on unseen test data[cite: 1].

---

## 🔑 Key Insights & Feature Importances
From the model interpretability analysis, the top factors driving car prices are[cite: 1]:
1. **Engine Size (`enginesize`):** The single most dominant factor determining base vehicle cost[cite: 1].
2. **Curb Weight (`curbweight`):** Strong positive correlation with build quality, safety structure, and price tier[cite: 1].
3. **Fuel Economy (`highwaympg`, `citympg`):** Inversely correlated with price (higher performance/luxury cars have lower MPG)[cite: 1].
4. **Power & Luxury Brand (`horsepower`, `Brand_bmw`, `Brand_porsche`):** Significant positive weight in premium segments[cite: 1].

---

## 🚗 Multi-Sample Prediction Testing (Unseen Test Data)

| Test Sample | Actual Price | Random Forest Predicted | Gradient Boosting Predicted |
| :--- | :---: | :---: | :---: |
| **Car #1** | $30,760.00 | $35,821.98 | $35,573.54 |
| **Car #2** | $17,859.17 | $19,619.61 | $18,976.17 |
| **Car #3** | $9,549.00 | $8,992.32 | $8,555.04 |
| **Car #4** | $11,850.00 | $13,186.97 | $13,227.72 |
| **Car #5** | $28,248.00 | $27,460.40 | $33,051.49 |

---

## 🔮 Live Interactive Ensemble Prediction
An interactive function `predict_combined_car_price()` takes real-time user specifications (Brand, Horsepower, Engine Size, Curb Weight, MPG, Body Style, Fuel Type) and calculates a **Recommended Market Valuation** using the ensemble average of both high-performing models[cite: 1]:

$$\text{Recommended Market Price} = \frac{\text{Price}_{\text{RandomForest}} + \text{Price}_{\text{GradientBoosting}}}{2}$$

```text
=================================================================
       ENTER CAR SPECIFICATIONS FOR FINAL PRICE ESTIMATE 
=================================================================
 • Specification : BMW (Convertible) | 200 HP | DIESEL
 • Fuel Economy  : 39 City / 45 Highway MPG
-----------------------------------------------------------------
 RECOMMENDED MARKET PRICE : $32,605.54
 Expected Valuation Range  : $31,299.09 – $33,911.99
=================================================================
