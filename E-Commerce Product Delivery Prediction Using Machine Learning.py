#!/usr/bin/env python
# coding: utf-8

# ## E-Commerce Product Delivery Prediction

# The project's goal is to develop a predictive model that determines whether a product from an e-commerce company will be delivered on time. It will also analyze different factors influencing delivery times and study customer behavior patterns related to the delivery process.
# 
# An international e-commerce firm specializing in electronics seeks to extract crucial insights from its customer database. They aim to employ sophisticated machine learning methods to analyze customer behavior and preferences.

# ### Data Dictionary 

# The dataset used for model building contained 10999 observations of 12 variables. The data contains:
# 
# | Variable | Description |
# | --- | --- |
# |ID|ID Number of Customers|
# |Warehouse_block|The Company have big Warehouse which is divided into block such as A,B,C,D,E|
# |Mode_of_Shipment|The Company Ships the products in multiple way such as Ship, Flight and Road|
# |Customer_care_calls|The number of calls made from enquiry for enquiry of the shipment|
# |Customer_rating|The company has rated from every customer. 1 is the lowest (Worst), 5 is the highest (Best)|
# |Cost_of_the_Product|Cost of the Product in US Dollars|
# |Prior_purchases|The Number of Prior Purchase|
# |Product_importance|The company has categorized the product in the various parameter such as low, medium, high|
# |Gender|Male and Female|
# |Discount_offered|Discount offered on that specific product|
# |Weight_in_gms|It is the weight in grams|
# |Reached.on.Time_Y.N|It is the target variable, where 1 Indicates that the product has NOT reached on time and 0 indicates it has reached on time|

# In[5]:


#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


#Loading the dataset
df = pd.read_csv('/Users/sowmya/Downloads/E_Commerce.csv')
df.head()


# ### Data Preprocessing I

# In[10]:


#Check the shape of the dataset
df.shape


# In[11]:


#Check data types of the columns
df.dtypes


# In[12]:


#Drop column
df.drop(['ID'], axis=1, inplace=True)


# In[13]:


#Check null/missing values
df.isnull().sum()


# In[14]:


#Check duplicate values
df.duplicated().sum()


# In[15]:


df.describe()


# In[16]:


df.head()


# ### Exploratory Data Analysis 

# During the exploratory data analysis, I will examine how the target variable interacts with other variables and analyze the distribution of these variables within the dataset to gain a deeper understanding of the data.

# #### Customer Gender Prediction

# In[17]:


plt.pie(df['Gender'].value_counts(),labels = ['F','M'], autopct='%1.1f%%', startangle=90)
plt.title('Gender Distribution')


# The dataset has the equal number of both males and female customers, with percentage of 49.6% and 50.4% respectively.

# #### Product Properties

# In[19]:


fig, ax = plt.subplots(1,3,figsize=(15,5))
sns.histplot(df['Weight_in_gms'], ax=ax[0], kde=True).set_title('Weight Distribution')
sns.countplot(x = 'Product_importance', data = df, ax=ax[1]).set_title('Product Importance')
sns.histplot(df['Cost_of_the_Product'], ax=ax[2], kde=True).set_title('Cost of the Product')


# The three graphs illustrate the distribution of product characteristics such as weight, cost, and importance within the dataset. The first graph shows that products primarily weigh between 1000-2000 grams and 4000-6000 grams, indicating that these weight categories are more prevalent in the company's sales. In the second graph, which represents product importance, we observe that most products are categorized as having low or medium importance. The third graph focuses on the cost distribution, highlighting a higher frequency of products priced between 150-200 and 225-275 dollars. 
# 
# Based on these observations, it's apparent that the company predominantly sells products weighing less than 6000 grams, with low to medium importance, and priced between 150-275 dollars.

# #### Logistics

# In[20]:


fig, ax = plt.subplots(1,3,figsize=(15,5))
sns.countplot(x = 'Warehouse_block', data = df, ax=ax[0]).set_title('Warehouse Block')
sns.countplot(x = 'Mode_of_Shipment', data = df, ax=ax[1]).set_title('Mode of Shipment')
sns.countplot(x = 'Reached.on.Time_Y.N', data = df, ax=ax[2]).set_title('Reached on Time')


# The graphs present insights into the logistics and delivery aspects of the products. The first graph indicates that warehouse F handles the most products, around 3500, while the other warehouses manage a comparable and lower number of products. The second graph displays the shipping methods, revealing that the majority of products are transported by ship, with about 2000 products shipped via flight and road. The third graph illustrates delivery timeliness, showing a higher quantity of products delivered on time compared to those that are late.
# 
# Considering these observations, it can be inferred that warehouse F might be strategically located near a seaport, as it not only has the highest volume of products but also predominantly uses shipping as the mode of transport.

# #### Customer Experience 

# In[21]:


fig, ax = plt.subplots(2,2,figsize=(15,10))
sns.countplot(x = 'Customer_care_calls', data = df, ax=ax[0,0]).set_title('Customer Care Calls')
sns.countplot(x = 'Customer_rating', data = df, ax=ax[0,1]).set_title('Customer Rating')
sns.countplot(x = 'Prior_purchases', data = df, ax=ax[1,0]).set_title('Prior Purchases')
sns.histplot(x = 'Discount_offered', data = df, ax=ax[1,1], kde = True).set_title('Discount Offered')


# The graphs provide an overview of customer experience metrics, including customer service interactions, ratings, previous purchases, and discounts. The first graph indicates that most customers make 3-4 customer care calls, suggesting possible issues with product delivery. The second graph shows an even distribution of customer ratings, with a slight increase in 1-star ratings, hinting at some level of dissatisfaction with the service.
# 
# The third graph reveals that a majority of customers have made 2-3 prior purchases, indicating that repeat customers are generally satisfied with the service and continue to engage with the company. The fourth graph displays the distribution of discounts, with most products receiving a 0-10% discount, suggesting that the company offers limited discounts on its products.

# #### Customer Gender and Product Delivery

# In[22]:


sns.countplot(x = 'Gender', data = df, hue = 'Reached.on.Time_Y.N').set_title('Gender vs Reached on Time')


# The data shows that the timely delivery of products is consistent across both genders, indicating that customer gender does not influence the punctuality of product delivery.

# #### Product Properties and Product Delivery

# In[23]:


fig, ax = plt.subplots(1,3,figsize=(15,5))
sns.violinplot(y = df['Weight_in_gms'], ax=ax[0], kde=True, x = df['Reached.on.Time_Y.N']).set_title('Weight Distribution')
sns.countplot(x = 'Product_importance', data = df, ax=ax[1], hue = 'Reached.on.Time_Y.N').set_title('Product Importance')
sns.violinplot(y = df['Cost_of_the_Product'], ax=ax[2], kde=True, x = df['Reached.on.Time_Y.N']).set_title('Cost of the Product')


# The plots illustrate how product characteristics affect delivery timeliness. The first graph reveals that product weight influences delivery punctuality; specifically, products weighing over 4500 grams tend to be delivered late, whereas those in the 2500-3500 gram range are more often delivered on time. The second graph, focusing on product importance, indicates that this factor does not significantly affect delivery timeliness. The third graph shows a correlation between product cost and delivery, with products priced above $250 experiencing more frequent delivery delays.
# 
# These observations suggest that both product weight and cost are significant factors affecting delivery timeliness.

# #### Logistics and Product Delivery

# In[25]:


fig, ax = plt.subplots(1,2,figsize=(15,5))
sns.countplot(x = 'Warehouse_block', data = df, ax=ax[0], hue = 'Reached.on.Time_Y.N').set_title('Warehouse Block')
sns.countplot(x = 'Mode_of_Shipment', data = df, ax=ax[1], hue = 'Reached.on.Time_Y.N').set_title('Mode of Shipment')


# The graphs demonstrate the connection between logistics operations and the timely delivery of products. Given that the majority of products are dispatched from warehouse F, which is presumed to be near a seaport due to its high shipping volume, it's notable that the mode of shipment is predominantly by ship.
# 
# However, the data shows a consistent difference in the number of products delivered on time versus late across all warehouses and shipping methods. This consistency suggests that the logistics, including the warehouse location and shipping method, do not significantly affect the timeliness of product delivery.

# #### Customer Experience and Product Delivery

# In[26]:


fig, ax = plt.subplots(2,2,figsize=(15,10))
sns.countplot(x = 'Customer_care_calls', data = df, ax=ax[0,0],hue = 'Reached.on.Time_Y.N').set_title('Customer Care Calls')
sns.countplot(x = 'Customer_rating', data = df, ax=ax[0,1],hue = 'Reached.on.Time_Y.N').set_title('Customer Rating')
sns.countplot(x = 'Prior_purchases', data = df, ax=ax[1,0],hue = 'Reached.on.Time_Y.N').set_title('Prior Purchases')
sns.violinplot(x = 'Reached.on.Time_Y.N', y = 'Discount_offered' ,data = df, ax=ax[1,1]).set_title('Discount Offered')


# The graphs link customer experience with product delivery for an E-Commerce company. The first graph shows that as customer care calls increase, on-time deliveries decrease, suggesting customers call more when deliveries are late. The second graph indicates that customers with higher ratings often receive their products on time. The third graph reveals that customers who make repeat purchases tend to receive their products on time, likely encouraging their continued business. Lastly, the fourth graph shows that products with less than 10% discount are often delivered late, while those with more than 10% discount are delivered on time more frequently.

# ### Data Preprocessing II

# In[27]:


from sklearn.preprocessing import LabelEncoder

#Label encoding object
le = LabelEncoder()

#columns for label encoding
cols = ['Warehouse_block','Mode_of_Shipment','Product_importance', 'Gender']

#label encoding
for i in cols:
    le.fit(df[i])
    df[i] = le.transform(df[i])
    print(i, df[i].unique())


# ### Correlation Matrix Heatmap

# In[28]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')


# The correlation matrix heatmap shows a positive correlation between the product's cost and the number of customer care calls.

# In[29]:


sns.violinplot(x = 'Customer_care_calls', y = 'Cost_of_the_Product', data = df)


# Customers tend to be more concerned about delivery when the product is expensive, leading to more customer service calls to check on the product's status. Therefore, ensuring timely delivery is crucial for high-cost items.

# ### Train Test Split

# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Reached.on.Time_Y.N', axis=1), df['Reached.on.Time_Y.N'], test_size=0.2, random_state=0)


# ### Model Building

# Using the following models to predict the product delivery:
# 
# - Random Forest Classifier
# 
# - Decision Tree Classifier
# 
# - Logistic Regression
# 
# - K Nearest Neighbors

# #### Random Forest Classifier

# In[33]:


from sklearn.ensemble import RandomForestClassifier

#Random Forest Classifier Object
rfc = RandomForestClassifier()


# In[34]:


#Using GridSearchCV for hyperparameter tuning
from sklearn.model_selection import GridSearchCV

#Parameter grid
param_grid = {
    'max_depth': [4,8,12,16],
    'min_samples_leaf': [2,4,6,8],
    'min_samples_split': [2,4,6,8],
    'criterion': ['gini', 'entropy'],
    'random_state': [0,42]
}

#GridSearchCV object
grid = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

#Fitting the model
grid.fit(X_train, y_train)

#Best parameters
print('Best parameters: ', grid.best_params_)


# In[35]:


#Random Forest Classifier Object
rfc = RandomForestClassifier(criterion='gini', max_depth=8, min_samples_leaf=8, min_samples_split=2, random_state=42)

#Fitting the model
rfc.fit(X_train, y_train)


# In[36]:


#Training accuracy
print('Training accuracy: ', rfc.score(X_train, y_train))


# In[37]:


#predicting the test set results
rfc_pred = rfc.predict(X_test)


# #### Decision Tree Classifier

# In[38]:


from sklearn.tree import DecisionTreeClassifier

#Decision Tree Classifier Object
dtc = DecisionTreeClassifier()


# In[39]:


#Using GridSearchCV for hyperparameter tuning
from sklearn.model_selection import GridSearchCV
#Parameter grid
param_grid = {
    'max_depth': [2,4,6,8],
    'min_samples_leaf': [2,4,6,8],
    'min_samples_split': [2,4,6,8],
    'criterion': ['gini', 'entropy'],
    'random_state': [0,42]}

#GridSearchCV object
grid = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

#Fitting the model
grid.fit(X_train, y_train)

#Best parameters
print('Best parameters: ', grid.best_params_)


# In[40]:


#Decision Tree Classifier Object
dtc = DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_leaf=6, min_samples_split=2, random_state=0, class_weight='balanced')

#Fitting the model
dtc.fit(X_train, y_train)


# In[41]:


#Training accuracy
print('Training accuracy: ', dtc.score(X_train, y_train))


# In[42]:


#predicting the test set results
dtc_pred = dtc.predict(X_test)


# #### Logistic Regression

# In[43]:


from sklearn.linear_model import LogisticRegression

#Logistic Regression Object
lr = LogisticRegression()


# In[44]:


#fitting the model
lr.fit(X_train, y_train)


# In[45]:


#Training accuracy
lr.score(X_train, y_train)


# In[46]:


#predicting the test set results
lr_pred = lr.predict(X_test)


# #### K Nearest Neighbors

# In[48]:


from sklearn.neighbors import KNeighborsClassifier

#KNN Classifier Object
knn = KNeighborsClassifier()


# In[49]:


#fitting the model
knn.fit(X_train, y_train)


# In[50]:


#training accuracy
knn.score(X_train, y_train)


# In[51]:


#predicting the test set results
knn_pred = knn.predict(X_test)


# ### Model Evaluation

# In[52]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, r2_score, mean_squared_error


# In[53]:


fig, ax = plt.subplots(2,2,figsize=(15,10))
sns.heatmap(confusion_matrix(y_test, rfc_pred), annot=True, cmap='coolwarm', ax=ax[0,0]).set_title('Random Forest Classifier')
sns.heatmap(confusion_matrix(y_test, dtc_pred), annot=True, cmap='coolwarm', ax=ax[0,1]).set_title('Decision Tree Classifier')
sns.heatmap(confusion_matrix(y_test, lr_pred), annot=True, cmap='coolwarm', ax=ax[1,0]).set_title('Logistic Regression')
sns.heatmap(confusion_matrix(y_test, knn_pred), annot=True, cmap='coolwarm', ax=ax[1,1]).set_title('KNN Classifier')


# In[54]:


#classification report
print('Random Forest Classifier: \n', classification_report(y_test, rfc_pred))
print('Decision Tree Classifier: \n', classification_report(y_test, dtc_pred))
print('Logistic Regression: \n', classification_report(y_test, lr_pred))
print('KNN Classifier: \n', classification_report(y_test, knn_pred))


# #### Model Comparison

# In[56]:


models = ['Random Forest Classifier', 'Decision Tree Classifier', 'Logistic Regression', 'KNN Classifier']
accuracy = [accuracy_score(y_test, rfc_pred), accuracy_score(y_test, dtc_pred), accuracy_score(y_test, lr_pred), accuracy_score(y_test, knn_pred)]
sns.barplot(x=models, y=accuracy, palette='magma').set_title('Model Comparison')
plt.xticks(rotation=90)
plt.ylabel('Accuracy')


# ### Conclusion

# The project's objective was to forecast on-time delivery for an e-commerce company's products and to explore factors influencing delivery times and customer behavior. The exploratory analysis highlighted that product weight and cost are crucial to delivery success, with products in the 2500-3500 gram range and priced under $250 being more likely to arrive on time. A significant volume of products was dispatched from warehouse F using shipping, suggesting its proximity to a seaport.
# 
# Customer behavior also sheds light on delivery outcomes. An increase in customer care calls often correlates with delivery delays. In contrast, customers with a history of multiple purchases tend to experience more punctual deliveries, which might explain their repeat business. As for discounts, products with minimal discounts (0-10%) saw more late deliveries, while those with discounts exceeding 10% were more often delivered on time.
# 
# Regarding machine learning models, the decision tree classifier outperformed others with a 69% accuracy rate. Close behind were the random forest classifier and logistic regression, with 68% and 67% accuracy, respectively. The K Nearest Neighbors model trailed with the least accuracy at 65%.
# 
