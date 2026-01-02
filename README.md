Student Performance Predictor:
- A machine learning project that predicts student performance index based on study habits and previous academic records.

Overview:
- This project uses Linear Regression to predict a student's performance index (0-100 scale) based on their study patterns, sleep habits, and extracurricular activities.
 
Dataset:
- Size: 10,000 student records (cleaned dataset)
- Features:
- Hours Studied
- Previous Scores
- Sleep Hours
- Sample Question Papers Practiced
- Extracurricular Activities
- Target Variable: Performance Index

Methodology:
- Data Preprocessing
- Applied StandardScaler for feature normalization
- One-hot encoding for categorical variables (Extracurricular Activities)

Model Training: 
- Algorithm: Linear Regression
- Train-Test Split: 80/20
 
Evaluation:
- Residual analysis to validate model assumptions

Results:
- R² Score: 0.989
- RMSE: 2.02
- Key Finding: Previous Scores are the strongest predictor (coefficient: 17.64)
- The model explains 98.9% of the variance in student performance, indicating excellent predictive power.

Feature Importance:
- Previous Scores (17.64)
- Hours Studied (7.39)
- Sleep Hours (0.81)
- Sample Papers Practiced (0.55)
- Extracurricular Activities (0.30)
  
Technologies Used:
- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
  
Visualizations:
- Feature importance (coefficient bar chart)
- Residual plot (validates linear regression assumptions)
  
How to Run:
- pip install pandas numpy scikit-learn matplotlib seaborn
- python student_performance.py
