# E-Commerce-Product-Delivery-Prediction
## Context
The company, specializing in electronic products, seeks insights from its customer database to optimize delivery performance and enhance customer satisfaction.

## Data Description
The dataset comprises 10999 observations across 12 variables, detailing customer interactions, product characteristics, and delivery outcomes. Key variables include:

- Warehouse block
- Mode of shipment
- Customer care calls
- Product cost
- Prior purchases
- Product importance
- Delivery performance (target variable)

## Methodology
**Data Preprocessing**: Cleaned and prepared data, handling missing values, duplicates, and irrelevant columns.

**Exploratory Data Analysis (EDA)**: Investigated distribution of variables, customer behavior, and logistics factors using visualizations.

**Feature Engineering**: Transformed categorical variables using label encoding.

**Model Building**: Deployed machine learning models like Random Forest, Decision Tree, Logistic Regression, and KNN to predict delivery outcomes.

**Model Evaluation**: Assessed models based on accuracy, confusion matrix, and classification reports.

## Key Insights
Product weight and cost significantly impact delivery timeliness.
Warehouse F, likely near a seaport, handles most shipments, predominantly via shipping.
Customer engagement (calls, prior purchases) and promotional discounts correlate with delivery performance.

## Models Performance
Decision Tree Classifier demonstrated the highest accuracy at 69%.
Random Forest and Logistic Regression showed comparable performance, with accuracies around 68% and 67%.
KNN had the lowest accuracy at 65%.

## Conclusion
The project highlights critical factors affecting product delivery timelines and offers a robust predictive model to aid the e-commerce company in streamlining its logistics operations.

