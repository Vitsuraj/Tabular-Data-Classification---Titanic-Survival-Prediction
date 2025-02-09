🚢 Tabular Data Classification - Titanic Survival Prediction
This repository contains a Jupyter Notebook for predicting passenger survival on the Titanic using machine learning techniques. The dataset is sourced from Kaggle's Titanic Competition.

📌 Project Overview
The goal of this project is to predict survival outcomes based on passenger features such as:

Age
Sex
Passenger Class
Fare, Embarked, Siblings/Spouses aboard, etc.
📂 Repository Structure
TATA_Titanic.ipynb → Jupyter Notebook containing data preprocessing, feature engineering, model training, and evaluation.
📊 Dataset
You can download the dataset from Kaggle's Titanic Competition.

🛠️ Tools & Libraries Used
Pandas, NumPy for data manipulation
Matplotlib, Seaborn for visualization
Scikit-Learn for machine learning models
🚀 How to Run
Clone the repository:
bash
Copy
Edit
git clone https://github.com/your-username/Tabular-Data-Classification-Titanic.git
Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Open and run TATA_Titanic.ipynb in Jupyter Notebook.
📈 Model Performance
The notebook explores multiple models, evaluates their accuracy, and selects the best-performing one.

🏆 Results
Key insights from data visualization
Feature importance analysis
Final model performance
📜 License
Machine Learning Models and Their Accuracy for Titanic Survival Prediction

In this project, multiple machine learning models were implemented to predict passenger survival on the Titanic dataset. The dataset was preprocessed by handling missing values, encoding categorical variables, and engineering new features such as FamilySize.

Models Used
Logistic Regression

A statistical model used for binary classification tasks.
Accuracy: Varies based on hyperparameters but typically around 78-80% on test data.
Random Forest Classifier

An ensemble learning method that uses multiple decision trees to improve prediction accuracy.
Accuracy: Approximately 81-85% on validation data.
Support Vector Machine (SVM)

A classification algorithm that tries to find the optimal hyperplane to separate classes.
Accuracy: Around 82-84% with hyperparameter tuning.
Model Evaluation
Accuracy scores were computed using accuracy_score and cross_val_score.
The Random Forest Classifier showed the best performance among the three models, achieving the highest accuracy.
Hyperparameter tuning was performed using GridSearchCV to optimize model performance.
