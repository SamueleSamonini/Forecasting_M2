# Forecasting US M2 Money Supply: Linear vs. Non-Linear Models

This project explores the predictability of the monthly growth rate of the US M2 Money Supply, comparing the efficacy of traditional linear econometric models against non-linear Machine Learning algorithms.

The analysis covers an extended time horizon (1959–2025), placing particular emphasis on model robustness during the liquidity shock caused by the COVID-19 pandemic in 2020.

## 🎯 Project Goal

To determine whether the complexity of non-linear models (ML) offers a predictive advantage over parsimonious linear models in a macroeconomic context characterized by *structural breaks* and high volatility.

## 📊 Data

Data is sourced from the **FRED** (Federal Reserve Bank of St. Louis) database:
* **Target:** `M2SL` (M2 Money Supply, Seasonally Adjusted).
* **Exogenous Regressor (for VAR):** `FEDFUNDS` (Effective Federal Funds Rate).
* **Range:** January 1959 – October 2025.
* **Transformations:** Data has been transformed via *Log-Difference* to ensure stationarity (Monthly Growth Rate).

## 🛠️ Methodology and Models

The study compares three modeling approaches using 1-step ahead *rolling forecasts*:

1.  **ARIMA (Univariate - Benchmark):**
    * Parameter selection $(p,d,q)$ via Grid Search minimizing AIC.
    * Optimal model identified: `ARIMA(4,0,3)`.
2.  **Vector Autoregression (VAR - Multivariate):**
    * Includes Interest Rates (`FEDFUNDS`) to test the impact of the cost of money on money supply.
    * Lag selection via AIC ($k=13$).
3.  **Random Forest (Machine Learning - Non-Linear):**
    * Supervised learning approach with 12 time-lag features.
    * Ensemble of 100 decision trees.

## 📂 Repository Structure

The code is organized into sequential Jupyter Notebooks to ensure reproducibility:

* `notebooks/`
    * `01_data_prep.ipynb`: Data loading, stationarity tests (ADF), transformations, and Train/Test split (80/20).
    * `02_modeling.ipynb`: ARIMA model estimation and Grid Search.
    * `03_ml_model.ipynb`: Feature engineering and Random Forest training.
    * `04_var_model.ipynb`: Multivariate dataset creation and VAR model estimation.
    * `05_evaluation.ipynb`: Final comparison, metrics calculation (RMSFE, MAFE), and plot generation.
* `data/`: Raw CSV files (`M2SL.csv`, `FEDFUNDS.csv`).
* `results/`: Comparison tables and exported plots.

## 📈 Key Results

The empirical analysis on the Test Set (2012-2025) yielded the following results:

| Model | Type | RMSFE | MAFE | Bias |
| :--- | :--- | :--- | :--- | :--- |
| **ARIMA (4,0,3)** | Linear / Univariate | **0.5455** | 0.2703 | -0.0168 |
| **VAR (13)** | Linear / Multivariate | 0.5462 | **0.2677** | -0.0143 |
| **Random Forest** | Non-Linear / ML | 0.6503 | 0.3038 | **0.0034** |

**Main Conclusions:**
* **Linear models (ARIMA and VAR)** dominate in terms of accuracy, successfully adapting to the 2020 shock due to the strong autoregressive inertia of the series.
* **Random Forest**, despite having the lowest Bias (unbiased on average), fails to capture volatility peaks due to the inability of decision trees to extrapolate values outside the training range (*bounding box problem*).
* Adding interest rates in the VAR model slightly improves the Mean Absolute Forecast Error (MAFE) but does not enhance outlier handling (RMSFE).

## 💻 Installation and Usage

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/m2-forecasting.git](https://github.com/your-username/m2-forecasting.git)
    ```
2.  Install required dependencies:
    ```bash
    pip install pandas numpy matplotlib scikit-learn statsmodels
    ```
3.  Run the notebooks in numerical order from `01` to `05`.

## 📜 License
This project is distributed under the MIT License.
