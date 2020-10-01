#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[ ]:


from glob import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt, pydotplus
get_ipython().run_line_magic('matplotlib', 'inline')


# # 1. Retrieving and Preparing the Data

# In[ ]:


data_store='C:/Users/pc/Desktop/data_model/Dataset'
sample_data = glob(data_store+"/*.csv")
sample_data[:14]


# This is just the list of all the data files we have in our Data store...there are 15 files.

# In[ ]:


def load_dataframe(sample_data):
    person = pd.DataFrame()
    for k,name in enumerate(sample_data):
        df = pd.read_csv(name, header=None) 
        df['Person_Id'] = k+1
    person = person.append(df.iloc[:,1:])
    return person
person_df = load_dataframe(sample_data)
person_df.columns = ['X-acc', 'Y-acc', 'Z-acc', 'Activity','Person_Id']
person_df.head(9)


# In[ ]:


print('Loaded %d subjects Successful' % len(person_df.Person_Id.unique()))


# In[ ]:


person_df.tail(10)


# In[ ]:


person_df.info()


# In[ ]:


person_df.isnull().sum()


# Its quite Amazing that we have no missing values 

# # Exploring the Columns

# X-acc variable statistics

# In[ ]:


person_df['X-acc'].min()


# In[ ]:


person_df['X-acc'].max()


# In[ ]:


person_df['X-acc'].mean()


# In[ ]:


person_df['X-acc'].median()


# In[ ]:


person_df['X-acc'].std()


# Y-acc variable statictics

# In[ ]:


person_df['Y-acc'].min()


# In[ ]:


person_df['Y-acc'].max()


# In[ ]:


person_df['Y-acc'].mean()


# In[ ]:


person_df['Y-acc'].median()


# In[ ]:


person_df['Y-acc'].std()


# Z-acc variable statisticts

# In[ ]:


person_df['Z-acc'].min()


# In[ ]:


person_df['Z-acc'].max()


# In[ ]:


person_df['Z-acc'].mean()


# In[ ]:


person_df['Z-acc'].median()


# In[ ]:


person_df['Z-acc'].std()


# In[ ]:


#The above ststistrics can be summarized as follows
person_df.describe()


# In[ ]:


person_df['Activity'].isnull().sum()


# Lets explore Activity per Person

# In[ ]:


for person in person_df.Person_Id:

        print (person, person_df[person_df.Person_Id == person].Activity)


# # 2. Data Exploration (EDA)

# # checking for Anomalies

# In[ ]:


#Plotting a box plot to test for abnomalities in dataset
sns.boxplot(data=person_df)


# In[ ]:


sns.boxplot(x=person_df['X-acc'])


# In[ ]:


print("MEAN VALUES ARE:")
person_df.mean()


# In[ ]:


sns.boxplot(x=person_df['Y-acc'])


# In[ ]:


sns.boxplot(x=person_df['Z-acc'])


# Both our data values for x.y and z are normaly distributed as see in the box plot..meaning majority of the values are close to the mean value for each attribute. Thats pretty fine  coz it means the data has no abnomalities that will affect our model.

# # Checking for duplicate values

# In[ ]:


# Total number of rows and columns
person_df.shape

# Rows containing duplicate data
duplicate_rows_person_df = person_df[person_df.duplicated()]
print("Dataframe shape is:",person_df.shape)
print("Number of duplicate rows:", duplicate_rows_person_df.shape)


# You can see there are 75611 raws out of 166743 raws with duplicate data.Lets check to see which raws are duplicated

# In[ ]:


person_df[person_df.duplicated()] # only show duplicates


# As you can see..Only the Lebel and the Person ID are duplicated....there is no need to remove them as it is expected as that

# # Attribute Correlation Using the person correlation methord

# In[ ]:


person_df.corr()


# In[ ]:


# We can make the vizualization of the corrrelation look more appealing by using a heatmap.

plt.figure(figsize=(20,10))
correlation= person_df.corr(method='pearson', min_periods=1)
sns.heatmap(correlation,cmap="BrBG",annot=True)
correlation


# In[ ]:


#plot the x, y, z acceleration and activities for a single person
def plot_person(person_df):
    pyplot.figure()
#create a plot for each column
    for col in range(person_df.shape[1]):
        pyplot.subplot(person_df.shape[1], 1, col+1)
        pyplot.plot(person_df[:,col])
        pyplot.show()
#plot activities for a single subject
    plot_person(person_df[person_df.Person_Id==1].iloc[:,:4].values)


# In[ ]:


subjects = []
for k,values in person_df.groupby('Person_Id'):
    subjects.append(values.iloc[:,:4].values)
# returns a list of dict, where each dict has one sequence per activity
def group_by_activity(subjects, activities):
    grouped = [{a:s[s[:,-1]==a] for a in activities} for s in subjects]
    return grouped
# calculate total duration in sec for each activity per subject and plot
def calculate_durations(grouped, activities):
# calculate the lengths for each activity for each subject
    freq = 52
    durations = [[len(s[a])/freq for s in grouped] for a in activities]
    return durations
def plot_durations(grouped, activities):
    durations = calculate_durations(grouped, activities)
    pyplot.boxplot(durations, labels=activities)
    pyplot.show()
# grouped
    activities = [i for i in range(0,8)]
    grouped = group_by_activity(subjects, activities)
# plot durations
    plot_durations(grouped, activities)


# # 3. Data Modelling

# # Defining the variables

# In[ ]:


X = person_df[['X-acc','Y-acc','Z-acc']]
y = person_df['Activity']


# # Train data split

# In[ ]:


#evaluate the model by splitting into train and test sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=12)


# # Modeling

# # Using K-Nearest Neighbor

# Moedel Fitting

# In[ ]:


##Import the Classifier.
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

##Instantiate the model with 5 neighbors.
model = KNeighborsClassifier(n_neighbors=5)

##Fit the model on the training data.
model.fit(X_train, y_train)


# Prediction

# In[ ]:


#use the model to make predictions with the test data
Y_pred = model.predict(X_test)

#how did our model perform?
count_misclassified = (y_test != Y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, Y_pred)
print('Accuracy: {:.2f}'.format(accuracy))


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy = []
for K in range(25):
    K_value = K+1
    neigh = KNeighborsClassifier(n_neighbors = K_value)
    neigh.fit(X_train, y_train)
    Y_pred = neigh.predict(X_test)
    accuracy.append(accuracy_score(y_test,Y_pred)*100)
    print("Accuracy is ", accuracy_score(y_test,Y_pred)*100,"% for K-Value:",K_value)


# Choosing a large value of K will lead to greater amount of execution time & under-fitting. Selecting the small value of K will lead to over-fitting.Thus,
# 
# There is no such guaranteed way to find the best value of K.

# # Using Decision Tree

# In[ ]:


# calculate cross-validated AUC
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# In[ ]:


# use the model to make predictions with the test data
prediction = model.predict(X_test)
# Import metrics
from sklearn import metrics
# generate evaluation metrics-
print(metrics.accuracy_score(y_test, prediction))


# In[ ]:


# Print out the confusion matrix
print(metrics.confusion_matrix(y_test, prediction))


# # Culculating error metrix

# In[ ]:


# Print out the classification report, and check the f1 score
print(metrics.classification_report(y_test, prediction))


# In[ ]:


from sklearn import tree, metrics, model_selection, preprocessing
from IPython.display import Image, display
import graphviz

dot_data = tree.export_graphviz(model,out_file=None,filled=True,rounded=True,)
graph = pydotplus.graph_from_dot_data(dot_data)
display(Image(graph.create_png()))
graph.write_png('decTreeOutput.png')


# # CONCLUSION
# Accuracy K_Nearest Neighbor is  84 % for K-Value: 25 While Decision Tree has an Acuracy of 77%.
#  

# 
# Since KNN Achievd the greatest accuracy, It is the most suitable Algorith to Predict the  HAR data

# The results could be improved by using more creativity in the feature

# In[ ]:





# In[ ]:




