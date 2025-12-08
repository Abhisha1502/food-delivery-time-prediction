
---

## 🚀 Project Objectives

### **1. Predict Delivery Time (Regression)**
Using Linear Regression, the model estimates delivery duration based on:
- Distance  
- Order Cost  
- Customer Rating  
- Delivery Partner Experience  
- Weather & Traffic Conditions  

### **2. Classify Whether Delivery Will Be Delayed (Classification)**
Logistic Regression is used to classify:
- **0 → Fast Delivery**
- **1 → Delayed Delivery**

---

## 🧼 Data Preprocessing

Key preprocessing steps:
- Handling missing values  
- Encoding categorical variables  
- Removing ID and text-based columns  
- Converting string fields into numeric formats  
- Standardizing structure for modeling  

Performed inside: `src/preprocessing.py`

---

## 🛠 Feature Engineering

Created new variables to improve model performance:
- **Delivery_Status** (binary classification target)  
- Extracted time components (optional)  
- Cleaned numerical and categorical fields  
- Prepared final feature sets for ML models  

Inside: `src/feature_engineering.py`

---

## 📊 Exploratory Data Analysis (EDA)

Conducted using:
- Histograms  
- Boxplots  
- Correlation heatmaps  

Revealed insights:
- Distance strongly correlates with Delivery Time  
- Order Cost and Delivery Time have moderate correlation  
- Experience slightly reduces delays  
- Delivery_Status aligns strongly with actual delivery time  
- Some outliers exist (Order Cost, Delivery Time)

Plots generated inside: `visuals/`

---

## 🤖 Machine Learning Models

### 🔹 **Linear Regression**
**Goal:** Predict delivery time (continuous values)

Metrics:
- MAE ≈ 6–8 minutes  
- RMSE ≈ 10–12 minutes  
- R² score indicates a reasonable predictive fit  

### 🔹 **Logistic Regression**
**Goal:** Predict delayed vs non-delayed deliveries  
Metrics:
- **Accuracy:** 0.875  
- **Precision:** 0.875  
- **Recall:** 1.00  
- **F1 Score:** 0.933  
- **Confusion Matrix** shows zero false negatives  
- **ROC Curve** indicates excellent separability  

---

## 📈 Key Visualizations

Generated and saved in `visuals/`:

| Visualization | Purpose |
|---------------|---------|
| Histograms | Understanding distributions |
| Boxplots | Detecting outliers |
| Correlation Heatmap | Feature relationships |
| Confusion Matrix | Classification evaluation |
| ROC Curve | Model separability |

---

## 🧠 Insights & Recommendations

- **Distance** is the strongest predictor of delivery time.  
- **Order Cost** impacts preparation time → longer wait.  
- **Traffic & Weather** influence delays → adjust ETAs dynamically.  
- **Delivery Experience** slightly improves performance.  
- Logistic model can be used for real-time delay alerts.  

---

## 🧪 How to Run the Notebook

1. Install dependencies:

```bash
pip install -r requirements.txt
