
# 🛡️ ChurnGuard — Customer Retention Intelligence

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-AA2222?style=for-the-badge)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> Predict which customers will churn — before they leave. Take targeted action. Save revenue.

---

## 🚀 Live Demo

🔗 [**Try it Live Here**](https://churnguard001.streamlit.app/)

---

## 💡 Problem Statement

E-commerce companies lose millions annually to customer churn. Traditional approaches react after customers leave — too late to act. **ChurnGuard identifies at-risk customers before they churn**, enabling retention teams to take targeted action at the right time.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎯 Single Prediction | Enter customer details → Get churn probability + risk level + retention recommendations |
| 📊 Batch Prediction | Upload CSV → Predict churn for all customers at once → Download results |
| 📈 Business Insights | Revenue at risk calculator + Tenure analysis + Top churn drivers |
| 🧠 Model Explainability | SHAP-based explanations — why will this customer churn? |

---

## 🔍 Key Findings

- **Customers with 0-6 months tenure** have **25.9% churn rate** vs only **1.3%** for 20+ month customers
- **First 6 months is the critical retention window** — invest heavily in onboarding
- **Complaint filed** is the strongest churn signal after tenure
- **Cashback alone does not retain customers** — loyalty programs more effective

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| XGBoost | Primary ML model (99.77% ROC-AUC) |
| SHAP | Model explainability — per-customer explanations |
| SMOTE | Handle class imbalance (83% vs 17%) |
| Scikit-learn | Model training + evaluation |
| Pandas + NumPy | Data cleaning + feature engineering |
| Plotly | Interactive visualizations |
| Streamlit | Dashboard + deployment |

---

## 📊 Model Performance

| Model | ROC-AUC | Accuracy | Churn Recall |
|-------|---------|----------|--------------|
| Logistic Regression | 0.8251 | 73% | 77% |
| Random Forest | 0.9952 | 98% | 92% |
| **XGBoost ✅** | **0.9977** | **98%** | **94%** |

> XGBoost correctly identifies **179 out of 190 actual churners** in the test set — missing only 11.

---

## 📸 Screenshots

## Single Prediction
<img width="1005" height="567" alt="image" src="https://github.com/user-attachments/assets/bf63baa1-acf9-44e7-b0ae-0ec7de6c0f71" />
<img width="1040" height="563" alt="image" src="https://github.com/user-attachments/assets/10c24072-6e30-4c69-9004-29387f08dab7" />

## Batch Prediction
<img width="1038" height="584" alt="image" src="https://github.com/user-attachments/assets/5385ccc9-8cd5-454f-a86f-be8db8783244" />
<img width="1019" height="590" alt="image" src="https://github.com/user-attachments/assets/37ab67c4-f3b5-4c85-86b0-ca1c71ed2547" />

## Buisness Insights
<img width="1008" height="537" alt="image" src="https://github.com/user-attachments/assets/ee4a1a83-470f-40e8-bab0-d66db6f77d22" />
<img width="1024" height="586" alt="image" src="https://github.com/user-attachments/assets/97820853-1bf5-4bf2-ba72-53a17e303c64" />
<img width="1061" height="586" alt="image" src="https://github.com/user-attachments/assets/17e80074-1d6b-42e1-b9ee-2abed0a56386" />

## Model Explainability
<img width="1048" height="594" alt="image" src="https://github.com/user-attachments/assets/abfa7587-01ff-4752-9b1f-ee9c91a34936" />
<img width="986" height="575" alt="image" src="https://github.com/user-attachments/assets/64c79fb8-4c78-4a1a-ab99-8f9edab4b94e" />

---

## ⚙️ Run Locally

```bash
git clone https://github.com/pritamk001/ChurnGuard.git
cd ChurnGuard
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔗 Links

- 🌐 **Live App:** [churnguard001.streamlit.app](https://churnguard001.streamlit.app/)
- 💻 **GitHub:** [github.com/pritamk001/ChurnGuard](https://github.com/pritamk001/ChurnGuard)

---

*Built with ❤️ using XGBoost + SHAP + Streamlit*
