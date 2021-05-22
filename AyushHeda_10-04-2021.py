#!/usr/bin/env python
# coding: utf-8

# # Problem 1 Clustering
# ## A leading bank wants to develop a customer segmentation to give promotional offers to its customers. They collected a sample that summarizes the activities of users during the past few months. You are given the task to identify the segments based on credit card usage.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

from sklearn.cluster import KMeans 

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix


# In[2]:


df = pd.read_csv("bank_marketing_part1_Data.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# # Checking Summary Statistics

# In[6]:


df.describe()


# # Checking for duplicates in the data

# In[7]:


df.duplicated().sum()


# #### There are no duplicates in the dataset

# In[8]:


df.head()


# # Ques 1

# ### Univariate, bivariate and multivariate analysis

# In[9]:


plt.figure(figsize = (20,10))
sns.boxplot(data = df)
plt.show()


# In[10]:


sns.pairplot(df)
plt.show()


# In[11]:


plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot = True)
plt.show()


# In[12]:


def univariateAnalysis_numeric(column,nbins):
    print("Description of " + column)
    print("----------------------------------------------------------------------------")
    print(df[column].describe(),end=' ')
    
    
    plt.figure()
    print("Distribution of " + column)
    print("----------------------------------------------------------------------------")
    sns.distplot(df[column], kde=True, color = 'g');
    plt.show()
    
    plt.figure()
    print("BoxPlot of " + column)
    print("----------------------------------------------------------------------------")
    sns.boxplot(x = df[column])
    plt.show()


# In[13]:


df_num = df.select_dtypes(include = ['float64', 'int64'])
lstnumericcolumns = list(df_num.columns.values)
len(lstnumericcolumns)


# In[14]:


for x in lstnumericcolumns:
    univariateAnalysis_numeric(x,20)


# # Scaling the Data

# In[15]:


X = StandardScaler()
scaled_df = pd.DataFrame(X.fit_transform(df.iloc[:,1:7]), columns = df.columns[1:])


# In[16]:


scaled_df.head()


# In[17]:


wardlink = linkage(scaled_df, method = 'ward')


# In[18]:


plt.figure(figsize=(10, 5))
dend = dendrogram(wardlink)
plt.show()


# In[19]:


dend = dendrogram(wardlink,
                 truncate_mode = 'lastp',
                 p = 8,
                 )


# In[20]:


clusters = fcluster(wardlink, 3, criterion='maxclust')
clusters


# In[21]:


df['clusters'] = clusters


# In[22]:


df.clusters.value_counts().sort_index()


# # Cluster Profiling

# In[23]:


aggdata=df.iloc[:,1:8].groupby('clusters').mean()
aggdata['Freq']=df.clusters.value_counts().sort_index()
aggdata


# In[24]:


df.head()


# # K - Means
# ## Creating Clusters using KMeans

# In[25]:


k_means = KMeans(n_clusters = 3,random_state = 0)


# In[26]:


k_means.fit(scaled_df)


# # Cluster Output for all the observations

# In[27]:


k_means.labels_


# In[28]:


k_means.inertia_


# # Calculating WSS for other values of K - Elbow Method

# In[29]:


wss = []


# In[30]:


for i in range(1,11):
    KM = KMeans(n_clusters = i, random_state = 1)
    KM.fit(scaled_df)
    wss.append(KM.inertia_)


# In[31]:


wss


# In[32]:


a=[1,2,3,4,5,6,7,8,9,10]


# In[33]:


sns.pointplot(a, wss)
plt.show()


# # KMeans with K=2

# In[34]:


k_means = KMeans(n_clusters = 2,random_state=0)
k_means.fit(scaled_df)
labels = k_means.labels_


# # Cluster evaluation for 2 clusters: the silhouette score

# In[35]:


from sklearn.metrics import silhouette_samples, silhouette_score


# In[36]:


silhouette_score(scaled_df,labels,random_state=0)


# In[37]:


df["Clus_kmeans4"] = labels
df.head()


# # Cluster Profiling

# In[38]:


df.Clus_kmeans4.value_counts().sort_index()


# In[39]:


clust_profile = df.groupby('Clus_kmeans4').mean()
clust_profile['freq'] = df.Clus_kmeans4.value_counts().sort_index()
clust_profile.T


# # Problem 2 CART-RF-ANN
# ## An Insurance firm providing tour insurance is facing higher claim frequency. The management decides to collect data from the past few years. You are assigned the task to make a model which predicts the claim status and provide recommendations to management. Use CART, RF & ANN and compare the models' performances in train and test sets.

# # CART

# In[40]:


dataset = pd.read_csv("insurance_part2_data.csv")


# In[41]:


dataset.head()


# In[42]:


dataset.describe()


# In[43]:


dataset.shape


# In[44]:


dataset.info()


# In[45]:


dataset.isnull().sum()


# In[46]:


plt.figure(figsize = (20,8))
sns.boxplot(data = dataset)
plt.show()


# In[47]:


def univariateAnalysis_numeric(column,nbins):
    print("Description of " + column)
    print("----------------------------------------------------------------------------")
    print(dataset[column].describe(),end=' ')
    
    
    plt.figure()
    print("Distribution of " + column)
    print("----------------------------------------------------------------------------")
    sns.distplot(dataset[column], kde=True, color = 'g');
    plt.show()
    
    plt.figure()
    
    print("BoxPlot of " + column)
    print("----------------------------------------------------------------------------")
    sns.boxplot(x = dataset[column])
    plt.show()


# In[48]:


dataset_num = dataset.select_dtypes(include = ['float64', 'int64'])
lstnumericcolumns = list(dataset_num.columns.values)
len(lstnumericcolumns)


# In[49]:


for x in lstnumericcolumns:
    univariateAnalysis_numeric(x,10)


# In[50]:


sns.pairplot(dataset)
plt.show()


# In[51]:


plt.figure(figsize=(10, 8))
corr = dataset.corr()
sns.heatmap(corr, annot = True)
plt.show()


# In[52]:


for feature in dataset.columns: 
    if dataset[feature].dtype == 'object': 
        dataset[feature] = pd.Categorical(dataset[feature]).codes


# In[53]:


X = dataset.drop("Claimed", axis=1)
y = dataset.pop("Claimed")


# In[54]:


from sklearn.model_selection import train_test_split

X_train, X_test, train_labels, test_labels = train_test_split(X, y, test_size=0.20, random_state = 0)


# In[55]:


dt_model = DecisionTreeClassifier(criterion = 'gini', random_state = 0)


# In[56]:


dt_model.fit(X_train, train_labels)


# In[57]:


from sklearn import tree

train_char_label = ['No', 'Yes']
Tree_File = open('dataset_tree.dot','w')
dot_data = tree.export_graphviz(dt_model, out_file=Tree_File, feature_names = list(X_train), class_names = list(train_char_label))

Tree_File.close()


# In[58]:


print (pd.DataFrame(dt_model.feature_importances_, columns = ["Imp"], index = X_train.columns))


# In[59]:


y_predict = dt_model.predict(X_test)


# # Regularizing the Decision Tree

# In[60]:


reg_dt_model = DecisionTreeClassifier(criterion = 'gini', max_depth = 7,min_samples_leaf=10,min_samples_split=30)
reg_dt_model.fit(X_train, train_labels)


# In[61]:


tree_regularized = open('tree_regularized.dot','w')
dot_data = tree.export_graphviz(reg_dt_model, out_file= tree_regularized , feature_names = list(X_train), class_names = list(train_char_label))

tree_regularized.close()

print (pd.DataFrame(dt_model.feature_importances_, columns = ["Imp"], index = X_train.columns))


# In[62]:


ytrain_predict = reg_dt_model.predict(X_train)
ytest_predict = reg_dt_model.predict(X_test)


# In[63]:


# AUC and ROC for the training data

# predict probabilities
probs = reg_dt_model.predict_proba(X_train)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(train_labels, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(train_labels, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()


# In[64]:


# AUC and ROC for the test data


# predict probabilities
probs = reg_dt_model.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(test_labels, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(test_labels, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()


# In[65]:


print(classification_report(train_labels, ytrain_predict))


# In[66]:


confusion_matrix(train_labels, ytrain_predict)


# In[67]:


reg_dt_model.score(X_train,train_labels)


# In[68]:


print(classification_report(test_labels, ytest_predict))


# In[69]:


confusion_matrix(test_labels, ytest_predict)


# In[70]:


reg_dt_model.score(X_test,test_labels)


# # Random Forest

# In[71]:


rfcl = RandomForestClassifier(n_estimators = 501)
rfcl = rfcl.fit(X_train, train_labels)


# In[72]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [7, 8, 9, 10],
    'min_samples_leaf': [15, 20, 25],
    'min_samples_split': [45, 60, 75],
    'n_estimators': [100, 300, 700] 
}

rfcl = RandomForestClassifier()

grid_search = GridSearchCV(estimator = rfcl, param_grid = param_grid, cv = 3)


# In[73]:


grid_search.fit(X_train, train_labels)


# In[74]:


grid_search.best_params_


# In[75]:


best_grid = grid_search.best_estimator_


# In[76]:


ytrain_predict = best_grid.predict(X_train)
ytest_predict = best_grid.predict(X_test)


# In[77]:


grid_search.score(X_train,train_labels)


# In[78]:


confusion_matrix(train_labels,ytrain_predict)


# In[79]:


print(classification_report(train_labels,ytrain_predict))


# In[80]:


print(classification_report(test_labels,ytest_predict))


# In[81]:


confusion_matrix(test_labels,ytest_predict)


# In[82]:


grid_search.score(X_test, test_labels)


# In[83]:


# AUC and ROC for the training data

# predict probabilities
probs = best_grid.predict_proba(X_train)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(train_labels, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(train_labels, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()


# In[84]:


# AUC and ROC for the test data


# predict probabilities
probs = best_grid.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(test_labels, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(test_labels, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()


# # ANN

# In[85]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# In[86]:


sc = StandardScaler() 
X_trains = sc.fit_transform(X_train) 
X_tests = sc.transform (X_test)


# In[87]:


clf = MLPClassifier(hidden_layer_sizes = 100, max_iter = 5000,
                     solver='lbfgs', verbose = True,  random_state = 0, tol = 0.01)
clf.fit(X_train, y_train)


# In[88]:


y_pred = clf.predict(X_test)


# In[89]:


accuracy_score(test_labels, y_pred) * 100


# In[90]:


param_grid = {
    'hidden_layer_sizes': [(100,200,300)],
    'activation': ['logistic', 'relu'],
    'solver': ['sgd', 'adam'],
    'tol': [0.1,0.001,0.0001],
    'max_iter' : [10000]
}

rfcl = MLPClassifier()

grid_search = GridSearchCV(estimator = rfcl, param_grid = param_grid, cv = 3)


# In[91]:


grid_search.fit(X_trains, train_labels)


# In[92]:


grid_search.best_params_


# In[93]:


best_grid = grid_search.best_estimator_


# In[94]:


ytrain_predict = best_grid.predict(X_trains)
ytest_predict = best_grid.predict(X_tests)


# In[95]:


confusion_matrix(train_labels,ytrain_predict)


# In[96]:


print(classification_report(train_labels,ytrain_predict))


# In[97]:


# AUC and ROC for the training data

# predict probabilities
probs = best_grid.predict_proba(X_train)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(train_labels, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(train_labels, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()


# In[98]:


accuracy_score(test_labels, ytest_predict) * 100


# In[99]:


confusion_matrix(test_labels,ytest_predict)


# In[100]:


print(classification_report(test_labels,ytest_predict))


# In[101]:


# AUC and ROC for the test data

# predict probabilities
probs = best_grid.predict_proba(X_test)
# keep probabilities for the positive outcome only88e
probs = probs[:, 1]
# calculate AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(test_labels, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(test_labels, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()


# # END
