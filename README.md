# Advanced Exploratory Data Analysis (EDA) Pipeline - Phase 3

An enterprise-grade data engineering and exploratory analysis pipeline designed to process real estate features, clean anomalous data, and extract statistical correlations.

## 📊 Pipeline Architecture
1. **Data Simulation Engine:** Generates structural records with stochastic noise, targeted missing features, and multi-variable outliers.
2. **Robust Data Cleansing:** Implements statistical imputation (Median-based tracking) for null distributions.
3. **Advanced Outlier Filtering:** Automates the Interquartile Range (IQR) method dynamically across feature blocks.
4. **Feature Correlation Matrix:** Computes Pearson correlation coefficients to isolate highly correlated target predictors.

## 🛠️ Key Libraries Used
* **Pandas:** For high-performance matrix scaling and structural grouping.
* **NumPy:** For vectorized execution of statistical bounds and deterministic random generation.

## 🚀 Execution & Performance Verification
To benchmark the analysis pipeline locally, run:
```bash
python main.py