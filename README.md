# Weekly Finance Collection - Payment Risk Prediction

### Machine Learning Project for Predicting Customer Payment Risk

---

## 1. Project Introduction

This project predicts the **Payment Risk** of customers who take weekly repayment loans.

The model classifies each customer’s weekly payment behaviour into three categories:

- **Low Risk**
- **Medium Risk**
- **High Risk**

This helps finance companies identify risky customers early and improve collection efficiency.

---

## 2. Dataset Details

**File Name:** `weekly_finance_collection_payment_behaviour.csv`

| Detail                  | Value          |
|-------------------------|----------------|
| Total Records           | 5,000          |
| Number of Columns       | 18             |
| Unique Customers        | Approximately 500 |
| Target Column           | Payment_Risk   |

### Target Distribution

| Payment Risk | Count | Percentage |
|--------------|-------|------------|
| Medium       | 2,742 | 54.8%      |
| High         | 1,343 | 26.9%      |
| Low          | 915   | 18.3%      |

---

## 3. Project Workflow

The complete process followed in this project:

1. **Load Dataset** – Read the CSV file using Pandas
2. **Data Cleaning** – Handle missing values in Payment_Date
3. **Feature Engineering** – Create new useful features
4. **Select Features & Target** – Prepare X and y
5. **Encode Target** – Convert Low/Medium/High into numbers
6. **Train-Test Split** – 80% training, 20% testing
7. **Train Models** – Train three different algorithms
8. **Evaluate Models** – Check Accuracy, F1-Score, Classification Report
9. **Select Best Model** – Choose the model with highest F1-Score
10. **Save Model** – Store the best model using Joblib

---

## 4. Feature Engineering

Four new features were created from the existing data:

| New Feature       | Formula / Logic                                      | Purpose                          |
|-------------------|------------------------------------------------------|----------------------------------|
| Payment_Ratio     | Amount_Paid ÷ Weekly_Due                             | Shows how much of due was paid   |
| Balance_Ratio     | Remaining_Balance ÷ Total_Payable                    | Shows remaining loan percentage  |
| Week_Progress     | Week_Number ÷ Total_Weeks                            | Shows loan progress              |
| Is_Partial        | True if partial payment was made                     | Detects incomplete payments      |

---

## 5. Algorithms Used

Three Machine Learning algorithms were trained and compared:

| Algorithm              | Type                        | Test Accuracy | F1-Score |
|------------------------|-----------------------------|---------------|----------|
| Logistic Regression    | Linear Classification       | 92.10%        | 0.9223   |
| Random Forest          | Ensemble (Bagging)         | **100%**      | **1.000**|
| Gradient Boosting      | Ensemble (Boosting)         | **100%**      | **1.000**|

**Best Performing Models:** Random Forest and Gradient Boosting

---

## 6. Features Used for Prediction (Total 16)

- Loan_Amount
- Interest_Rate_Percent
- Interest_Amount
- Total_Payable
- Weekly_Due
- Total_Weeks
- Week_Number
- Amount_Paid
- Days_Late
- Remaining_Balance
- Previous_Late_Count
- Missed_Payments
- Payment_Ratio
- Balance_Ratio
- Week_Progress
- Is_Partial

---

## 7. How to Run the Project

### Step 1: Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/d7ec1922-29c5-43da-af07-0cc0ac93b0cb" />
