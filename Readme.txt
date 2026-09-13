Student Placement Prediction

A machine learning project that predicts whether a student is likely to get placed based on academic performance, technical skills, experience, and other student profile features.

The project is built using Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn. I built the project from the beginning to understand the complete supervised learning workflow: loading the data, understanding the dataset, performing EDA, preparing features, training a Logistic Regression model, and evaluating the results.

Project Overview

The goal of this project is to predict:

0 → Not Placed
1 → Placed

The dataset contains 100,000 student records and multiple academic, technical, and extracurricular features.

The main idea is to use information that would be available about a student before placement and predict the placement outcome.

Dataset

The dataset contains features such as:

branch
college_tier
cgpa
backlogs
coding_skills
dsa_score
aptitude_score
communication_skills
ml_knowledge
system_design
internships
projects_count
certifications
hackathons
open_source_contributions
extracurriculars
placement_status
salary_package_lpa

placement_status is the target variable.

Why salary_package_lpa is not used

salary_package_lpa is not used as an input feature because salary is information that becomes available after/during the placement outcome.

Using salary to predict whether someone gets placed would cause data leakage.

For example:

Student information
       ↓
Predict placement
       ↓
Placement happens
       ↓
Salary is known

Therefore, salary should not be available to the model when making the placement prediction.

Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook
Project Workflow

The project follows this workflow:

Load Dataset
     ↓
Understand Dataset
     ↓
Check Missing Values
     ↓
Check Duplicates
     ↓
Exploratory Data Analysis
     ↓
Feature / Target Selection
     ↓
Train-Test Split
     ↓
Preprocessing
     ↓
Logistic Regression
     ↓
Predictions
     ↓
Model Evaluation
1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
Pandas

Pandas is used for working with tabular data.

import pandas as pd

The pd name is an alias for Pandas.

It allows us to use functions such as:

pd.read_csv()
NumPy
import numpy as np

NumPy is used for numerical operations and works closely with Pandas and Scikit-learn.

Matplotlib
import matplotlib.pyplot as plt

Matplotlib is used to create visualizations such as:

Histograms
Bar charts
Boxplots
Decision boundary plots

plt is the common alias for matplotlib.pyplot.

Seaborn
import seaborn as sns

Seaborn is useful for statistical visualizations such as correlation heatmaps.

2. Load the Dataset
df = pd.read_csv("dataset.csv")

pd.read_csv() reads the CSV file and converts it into a Pandas DataFrame.

df is the variable containing our dataset.

A DataFrame is basically a table containing rows and columns.

3. View the Dataset

To see the first five rows:

df.head()

head() is useful for quickly checking:

Column names
Values
Data format
Whether the dataset loaded correctly

We can also see more rows:

df.head(10)
4. Check Dataset Size
df.shape

shape returns:

(number of rows, number of columns)

This dataset contains 100,000 student records.

5. Check Data Types
df.dtypes

This tells us the type of data stored in each column.

Common types include:

int64    → whole numbers
float64  → decimal numbers
object   → usually text/categorical data

We can also use:

df.info()

info() gives a summary containing:

Number of rows
Column names
Data types
Non-null values
6. Find Numerical Columns
df.select_dtypes(include="number").columns

This returns columns containing numerical values.

Numerical features include things such as:

cgpa
coding_skills
dsa_score
internships
projects_count
7. Find Categorical Columns
df.select_dtypes(include="object").columns

This finds columns containing categorical/text values.

For this dataset, examples include:

branch
college_tier

Categorical features will later need to be encoded before being used by Logistic Regression.

8. Check Missing Values
df.isnull()

This checks every cell and returns:

True  → missing
False → not missing

However, this creates a large True/False table.

A more useful version is:

df.isnull().sum()

This counts missing values in every column.

The dataset contains missing values in:

salary_package_lpa

These missing salary values correspond to students who were not placed.

Since salary is not used as a feature, we don't need to fill these values for the placement model.

9. Check Duplicate Rows
df.duplicated().sum()

duplicated() identifies duplicate rows.

sum() counts them.

The dataset contains:

0 duplicate rows
10. Understand the Target Variable

The target column is:

df["placement_status"]

The values mean:

0 → Not Placed
1 → Placed

To count each class:

df["placement_status"].value_counts()

The dataset contains:

Not Placed → 31,525
Placed     → 68,475

So the target classes are not perfectly balanced.

11. Visualize Placement Distribution
df["placement_status"].value_counts().plot(kind="bar")

plt.xlabel("Placement Status")
plt.ylabel("Number of Students")
plt.title("Placement Status Distribution")

plt.show()

This creates a bar chart showing how many students belong to each class.

This is useful because we can immediately see whether the dataset has a class imbalance.

Exploratory Data Analysis

EDA stands for Exploratory Data Analysis.

The purpose of EDA is to understand the data before building the machine learning model.

Instead of immediately training a model, we first ask questions about the data.

12. Analyze CGPA
df["cgpa"].describe()

describe() gives statistics such as:

Count
Mean
Standard deviation
Minimum
25th percentile
Median
75th percentile
Maximum

We also created a histogram:

plt.hist(df["cgpa"])

plt.xlabel("CGPA")
plt.ylabel("Number of Students")
plt.title("Distribution of CGPA")

plt.show()

A histogram helps us understand how CGPA values are distributed.

13. CGPA vs Placement

We compared average CGPA between placed and non-placed students:

df.groupby("placement_status")["cgpa"].mean()

The results were approximately:

Not Placed → 7.003
Placed     → 7.300

This indicates that placed students have a higher average CGPA in this dataset.

However, this does not mean that higher CGPA guarantees placement.

A better conclusion is:

There is an observed positive association between CGPA and placement status.

14. CGPA Distribution by Placement

We also used a boxplot:

df.boxplot(column="cgpa", by="placement_status")

plt.xlabel("Placement Status")
plt.ylabel("CGPA")
plt.title("CGPA Distribution by Placement Status")

plt.suptitle("")

plt.show()

A boxplot helps us understand:

Median
Spread
Distribution
Possible outliers

for each placement group.

15. Placement Rate by CGPA

To understand the relationship more clearly, CGPA was divided into groups:

df["cgpa_group"] = pd.cut(
    df["cgpa"],
    bins=[0, 6, 7, 8, 9, 10],
    labels=["<6", "6-7", "7-8", "8-9", "9-10"]
)

Then we calculated placement rate:

df.groupby("cgpa_group", observed=True)["placement_status"].mean()

The observed placement rates were approximately:

<6   → 56.40%
6-7  → 63.46%
7-8  → 70.83%
8-9  → 77.13%
9-10 → 83.50%

This shows a clear positive relationship between CGPA and placement rate.

The cgpa_group column was created only for EDA and is not used as a final model feature.

16. Coding Skills Analysis

We checked the distribution:

df["coding_skills"].describe()

The coding skill score ranges from 1 to 10, with an average close to 6.

Then we compared it by placement status:

df.groupby("placement_status")["coding_skills"].mean()

The results were approximately:

Not Placed → 5.80
Placed     → 6.08

This suggests a positive association between coding skills and placement, although the difference is relatively small.

17. Internship Analysis

Instead of only comparing average internships, we calculated the placement rate for each internship count:

df.groupby("internships")["placement_status"].mean()

The results were:

0 internships → 62.76%
1 internship  → 68.28%
2 internships → 73.18%
3 internships → 77.18%

This shows that placement rate increases as internship experience increases in this dataset.

We also compared average internships:

df.groupby("placement_status")["internships"].mean()

Approximately:

Not Placed → 0.96
Placed     → 1.16

Again, this shows association, not causation.

18. DSA, Aptitude and Communication

We compared three features together:

df.groupby("placement_status")[
    ["dsa_score", "aptitude_score", "communication_skills"]
].mean()

Results:

Feature	Not Placed	Placed
DSA Score	5.27	5.61
Aptitude Score	64.19	65.36
Communication Skills	5.88	6.04

All three features have slightly higher averages among placed students.

This suggests they may contain useful information for the prediction model.

19. Correlation Analysis

We calculated correlations between numerical features and placement:

df.select_dtypes(include="number").corr()["placement_status"].sort_values()

The strongest positive correlations with placement were approximately:

Feature	Correlation
cgpa	0.149
internships	0.100
coding_skills	0.088
dsa_score	0.087
projects_count	0.070
certifications	0.056
communication_skills	0.051
aptitude_score	0.045

backlogs had a small negative correlation:

backlogs → -0.059

The correlations are relatively small, which tells us that placement cannot be explained by one numerical feature alone.

This is why we use multiple features in the machine learning model.

Important

Correlation does not mean causation.

It also does not mean that the feature with the highest correlation will automatically be the most important feature in the final Logistic Regression model.

20. Categorical Analysis

The dataset contains categorical variables such as:

branch
college_tier

We can calculate placement rates using:

df.groupby("branch")["placement_status"].mean()

and:

df.groupby("college_tier")["placement_status"].mean()

This helps us understand whether placement rates differ between different branches and college tiers.

These variables will later be converted into numerical representations using One-Hot Encoding.

21. Feature and Target Selection

The target variable is:

y = df["placement_status"]

The features are the information we use to make the prediction.

We do not use:

placement_status
salary_package_lpa
cgpa_group

as model features.

So:

X = df.drop(
    columns=["placement_status", "salary_package_lpa", "cgpa_group"]
)

y = df["placement_status"]

Here:

X → input features
y → target/output
22. Train-Test Split

Before training the model, we split the data:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=42,
    stratify=y
)
Why split the data?

If we train and test on exactly the same data, we won't know whether the model can generalize to new students.

Therefore:

90% → Training
10% → Testing

The model learns from the training data and is evaluated on unseen test data.

random_state=42

Makes the split reproducible.

stratify=y

Keeps the placement class proportions approximately the same in both training and testing datasets.

23. Preprocessing

Our dataset contains both numerical and categorical features.

Numerical features may have different scales.

For example:

CGPA            → 5-10
Aptitude score  → 0-100
Internships     → 0-3

Categorical columns contain values such as branch or college tier.

Therefore, we need different preprocessing for different columns.

We use:

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
StandardScaler

Standardizes numerical features.

OneHotEncoder

Converts categorical values into numerical columns.

ColumnTransformer

Allows us to apply different transformations to different columns.

24. Logistic Regression

The main model used in this project is Logistic Regression.

from sklearn.linear_model import LogisticRegression

Logistic Regression is a supervised learning algorithm commonly used for binary classification.

Our problem has two classes:

0 → Not Placed
1 → Placed

The model estimates the probability that a student belongs to class 1.

For example:

0.85 → approximately 85% predicted probability of placement

The probability is then converted into a class prediction.

25. Pipeline

Instead of manually preprocessing the data and then training the model separately, we can combine the steps using a Scikit-learn Pipeline.

from sklearn.pipeline import Pipeline

A pipeline can look like:

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

The pipeline makes sure that preprocessing happens correctly before the model is trained.

It also reduces the risk of accidentally fitting preprocessing steps using test data.

26. Train the Model

The model is trained using:

model.fit(X_train, y_train)

fit() means the model learns patterns from the training data.

The model learns coefficients associated with the input features that help separate placed and non-placed students.

27. Make Predictions

After training:

y_pred = model.predict(X_test)

This generates predicted classes:

0 → predicted not placed
1 → predicted placed

We can also obtain probabilities:

y_prob = model.predict_proba(X_test)[:, 1]

predict_proba() gives probabilities for each class.

[:, 1] selects the probability of class 1, which represents placement.

28. Evaluate the Model

Accuracy:

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

Accuracy measures:

Correct Predictions
-------------------
Total Predictions

However, because our dataset is not perfectly balanced, accuracy should not be the only metric.

We should also look at:

Precision
Recall
F1-score
Confusion Matrix
29. Classification Report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

The classification report gives:

Precision
Recall
F1-score
Support
Precision

Of the students predicted as placed, how many were actually placed?

Recall

Of the students who were actually placed, how many did the model correctly identify?

F1-score

A combined metric based on precision and recall.

30. Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(
    confusion_matrix=cm
).plot()

plt.show()

The confusion matrix contains:

True Positive
True Negative
False Positive
False Negative

For this project:

True Positive
→ Predicted placed and actually placed

True Negative
→ Predicted not placed and actually not placed

False Positive
→ Predicted placed but actually not placed

False Negative
→ Predicted not placed but actually placed

The confusion matrix helps us understand the types of mistakes made by the model.

31. Decision Boundary

The original project idea includes visualizing a Logistic Regression decision boundary.

A simple decision boundary can be shown when using two numerical features.

However, the final model uses many features and categorical variables, so a 2D decision boundary cannot represent the complete final model.

A two-feature decision boundary can still be used as a learning visualization to understand how Logistic Regression separates two classes.

What I Learned

This project helped me understand the complete supervised learning workflow.

Python
Variables
Functions
Imports
Basic data handling
Pandas
DataFrames
Reading CSV files
Selecting columns
head()
shape
dtypes
info()
isnull()
isnull().sum()
duplicated()
groupby()
mean()
describe()
EDA
Target distribution
Histograms
Bar charts
Boxplots
Group analysis
Placement rate
Correlation
Interpreting relationships
Machine Learning
Features and target
Train-test split
Stratification
Standardization
One-hot encoding
ColumnTransformer
Pipeline
Logistic Regression
Predictions
Probabilities
Classification metrics
Confusion matrix
Important ML concepts
Data leakage
Class imbalance
Generalization
Feature preprocessing
Model evaluation
Correlation vs causation
Project Structure
student-placement-prediction/
│
├── student_placement_prediction.ipynb
├── dataset.csv
└── README.md

The notebook contains the complete workflow from loading the dataset to model evaluation.

How to Run
1. Clone the repository
git clone <your-repository-url>
2. Install the required libraries
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
3. Open Jupyter Notebook
jupyter notebook
4. Open
student_placement_prediction.ipynb
5. Run the notebook from top to bottom

Make sure the dataset path in:

pd.read_csv(...)

matches the location of your CSV file.

Final Summary

This project predicts student placement outcomes using Logistic Regression.

The project covers the complete machine learning workflow:

Data Collection
      ↓
Data Understanding
      ↓
Data Cleaning
      ↓
EDA
      ↓
Feature Selection
      ↓
Train-Test Split
      ↓
Preprocessing
      ↓
Logistic Regression
      ↓
Prediction
      ↓
Evaluation

The EDA showed that CGPA, internships, coding skills, DSA, projects, certifications, communication skills, and aptitude have varying levels of association with placement.

Instead of relying on one variable, the final model uses multiple student characteristics together to make the prediction.

Note: This project is for learning and portfolio purposes. A real-world placement prediction system would require additional validation, fairness checks, monitoring, and testing on data from a different time period or student population before being used for actual placement decisions.
