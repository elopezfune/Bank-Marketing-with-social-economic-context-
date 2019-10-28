#!/usr/bin/env python
# coding: utf-8

# # Title: Bank Marketing (with social/economic context)
Cite as: [Moro et al., 2014] 
S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, In press, http://dx.doi.org/10.1016/j.dss.2014.03.001

Available at: [pdf] http://dx.doi.org/10.1016/j.dss.2014.03.001

              [bib] http://www3.dsi.uminho.pt/pcortez/bib/2014-dss.txt

Created by: Sérgio Moro (ISCTE-IUL), Paulo Cortez (Univ. Minho) and Paulo Rita (ISCTE-IUL) @ 2014
Past Usage: The full dataset (bank-additional-full.csv) was described and analyzed in:
S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems (2014), doi:10.1016/j.dss.2014.03.001.

 
Relevant Information:

1) This dataset is based on "Bank Marketing" UCI dataset (please check the description at: http://archive.ics.uci.edu/ml/datasets/Bank+Marketing).

2) The data is enriched by the addition of five new social and economic features/attributes (national wide indicators from a ~10M population country), published by the Banco de Portugal and publicly available at: https://www.bportugal.pt/estatisticasweb.

3) This dataset is almost identical to the one used in [Moro et al., 2014] (it does not include all attributes due to privacy concerns). 

4) Using the rminer package and R tool (http://cran.r-project.org/web/packages/rminer/), we found that the addition of the five new social and economic attributes (made available here) lead to substantial improvement in the prediction of a success, even when the duration of the call is not included. Note: the file can be read in R using: d=read.table("bank-additional-full.csv",header=TRUE,sep=";")

   
The zip file includes two datasets: 
1) bank-additional-full.csv with all examples, ordered by date (from May 2008 to November 2010).
2) bank-additional.csv with 10% of the examples (4119), randomly selected from bank-additional-full.csv.
The smallest dataset is provided to test more computationally demanding machine learning algorithms (e.g., SVM).

The binary classification goal is to predict if the client will subscribe a bank term deposit (variable y).
# # Initializing and importing modules for exploratory analysis
# 
# In this section we load the main Python libraries for database manipulation and visualization. These are the libraries: Numpy, Pandas, Matplotlib and Seaborn.

# In[1]:


import numpy as np   #Importing Numpy
import pandas as pd  #Importing Pandas

#Data visualization
import matplotlib    #Importing Matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rc('font', size=16)                #Use big fonts and big plots
plt.rcParams['figure.figsize'] = (10.0,10.0)    
matplotlib.rc('figure', facecolor='white')

import seaborn as sns #Importing Seaborn


# # Loading the database for exploratory analysis
# 
# We use the Pandas library to visualize the first rows and the columns of the database under study. The database is stored in the same folder as the project, to we only have to call it using the basic command pandas.read_csv("path/to/file"). We print the header of the database, showing the first 5 rows and the columns.

# In[2]:


dataframe = pd.read_csv('bank.csv') #Importing the database
dataframe.head() #Visualize the first 5 rows and the colunms of the database

After printing the firsts rows and columns of the database, we could start identifying the variables.

Input variables or bank client data:

1 - age (numerical variable)
2 - job : type of job (categorical variable: "admin.","blue-       collar","entrepreneur","housemaid","management","retired","self- employed","services","student","technician","unemployed","unknown")
3 - marital : marital status (categorical variable: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)
4 - education (categorical variable: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown")
5 - default: has credit in default? (categorical variable: "no","yes","unknown")
6 - housing: has housing loan? (categorical variable: "no","yes","unknown")
7 - loan: has personal loan? (categorical variable: "no","yes","unknown")

# related with the last contact of the current campaign:
8 - contact: contact communication type (categorical variable: "cellular","telephone") 
9 - month: last contact month of year (categorical variable: "jan", "feb", "mar", ..., "nov", "dec")
10 - day_of_week: last contact day of the week (categorical variable: "mon","tue","wed","thu","fri")
11 - duration: last contact duration, in seconds (numerical variable). Important note: this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

   # other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numerical variable, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numerical variable; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numerical variable)
15 - poutcome: outcome of the previous marketing campaign (categorical variable: "failure","nonexistent","success")

   # social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numerical variable)
17 - cons.price.idx: consumer price index - monthly indicator (numerical variable)     
18 - cons.conf.idx: consumer confidence index - monthly indicator (numerical variable)     
19 - euribor3m: euribor 3 month rate - daily indicator (numerical variable)
20 - nr.employed: number of employees - quarterly indicator (numerical variable)

Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary variable: "yes","no")

# # Exploratory analysis
# 
# First we are going to check if there are missing values within the columns of the database. To that end, we build the function "check_missing_values" that takes as argument the database and gives as a result, the list of the columns with the missing percent of the values, let's say, the appearance frequency of the NaN symbol.

# In[3]:


def check_missing_values(df,cols=None,axis=0):
    ### This function check out for missing values in each column
    ## Arguments:
                #df: data frame
                #cols: list. List of column names
                #axis: int. 0 means column and 1 means row
    
    # This function returns the missing info as a dataframe 
    
    if cols != None:
        df = df[cols]
    missing_num = df.isnull().sum(axis).to_frame().rename(columns={0:'missing_num'})
    missing_num['missing_percent'] = df.isnull().mean(axis)*100
    return missing_num.sort_values(by='missing_percent',ascending = False) 


print("We can see that there are no missing values within the database.")
check_missing_values(dataframe,cols=None,axis=0)

The goal of the analysis is to check if the clients have subscribed a term deposit. This is measured by the variable "y" in the last column of the database. We want to explore how many clients have subscribed a term deposit and to check if the data is well balanced or not. For that, we just have to count how many clients have subscribed the deposit or not. We normalize the counts to give in form of percent. 
# In[4]:


loan = dataframe["y"].value_counts(normalize=True).to_frame()
plt.rcParams['figure.figsize'] = (10.0, 4.0)    # ... and big plots
loan.plot.bar(color='g')
loan.T

We can see that only around 11.26% of the total cases subscribed a deposit. The data is not well balanced as there is a miss-match between the different cases.As said in the introduction, there a bunch of categorical variables that give a qualitative picture of the problem and they could be used to get hints and find patterns in data. 

The categorical variables are:
1) job
2) marital
3) education
4) default
5) housing
6) loan
7) contact
8) month
9) day_of_week
10) poutcome

We are going to build a function "compare_category" that takes as arguments the database we are working on, the name of the categorical variable we want to analyze "colname" and the output variable "targetname". The outgoing result of this function is a graphic counting the appearance frequency of each element within "colname" and a line describing the mean value of each element taking the value 1.
# In[5]:


### We can see that there are many cathegorical variables in the databases what could teach us about
### what is going on with the frauds.
### We are going to build a function that graphs which of these categorical variables have more in common
### with the credit card frauds.

def compare_category(df,colname,targetname):
    ### This function checks the target value difference of a given cathegory in the case
    ### of binary classifications.
    
    ## Arguments:
    # df: is a data frame.
    # colname: is a string. The column name to be evaluated.
    # targetname: is a string. The column name of the target variable.
    
    # caculate aggregate stats
    df_cate = df.groupby([colname])[targetname].agg(['count', 'sum', 'mean'])
    df_cate.reset_index(inplace=True)
    #print(df_cate)
    
    # plot visuals
    f, ax = plt.subplots(figsize=(20, 8))
    plt1 = sns.lineplot(x=colname, y="mean", data=df_cate,color="b")
    plt.xticks(size=20,rotation=90)
    plt.yticks(size=20,rotation=0)
    
    for tl in ax.get_yticklabels():
        tl.set_color('b')

    ax2 = ax.twinx()
    plt2 = sns.barplot(x=colname, y="count", data=df_cate,
                       ax=ax2,alpha=0.5)   
    
    
#MERGING BOTH DATABASES FOR EASIER ANALYSIS
#dataset_train_identity_transaction = pd.concat([dataset_train_identity,dataset_train_transaction],sort=True)


# In[6]:


cat_vars = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]
for el in cat_vars:
    compare_category(dataframe,el,"y")

We could see that:
job: Most of the clients are admin and blue-collar, but the most granted deposits are for students and retireds.
marital: Most of the clients are married, but the most granted deposits are for unknonwn marital status.
education: Most of the clients have a university degree, but the most granted deposits are for iliterates.
default: Most of the clients don't have credit in default, but they have the most granted deposits.
housing: Most of the clients have a house on loan and they have the most granted deposits.
loan: Most of the clients have granted loans and therefore granted deposits.
contact: Most of the clients have contact communication via cellphone, those have granted deposits.
month: May and July the month that clients ask for deposit, they are granted in March and December.
day_of_week: Monday and Thursday are more frequents, the grants are more frequent Thursday, Tuesday 
poutcome: more "inexistant"voutcome of the previous marketing campaign, but mostly success the grants.The minimum age asking for a deposit is 17 years and the maximum is 98; how ever, the age where the most granted deposits appear are around 98, 83, 36 and 78; as can be seen in the graphic below. Below 34 years old, no deposits exist.
# In[7]:


compare_category(dataframe,"age","y")

We are going to replace: admin., blue-collar, entrepreneur, technician, services, management and self-employed by "employed"; while the categories: retired, housemaid and student, we are going to replace by "unemployed". We will have in total three categories: [employed, unemployed, unknown].
# In[8]:


#dataframe['job'] = dataframe['job'].replace(["admin.", "blue-collar", "entrepreneur", "technician", "services", "management", "self-employed"], 'employed')
#dataframe['job'] = dataframe['job'].replace(["retired", "housemaid", "student"], 'unemployed')


# In[9]:


#dataframe['job'].value_counts()

We are going to replace: university.degree, high.school, basic.9y, professional.course, basic.4y and basic.6y  by "literate". We will have in total three categories: [literate, illiterate, unknown].
# In[10]:


#dataframe["education"] = dataframe["education"].replace(["university.degree", "high.school", "basic.9y", "professional.course", "basic.4y", "basic.6y"],"literate")


# In[11]:


#dataframe["education"].value_counts()


# In[12]:


#dataframe

We have to numerize all the categorical variables, for that I will use the following convention:
# In[13]:


dataframe=pd.get_dummies(dataframe)


# In[14]:


#dataframe['job'] = dataframe['job'].map( {"employed":1, "unemployed":0, "unknown":-1 })
#dataframe['marital'] = dataframe['marital'].map( {"married":2, "single":1, "divorced":0, "unknown":-1 })
#dataframe['education'] = dataframe['education'].map({"literate":1, "illiterate":0, "unknown":-1})
#dataframe['default'] = dataframe['default'].map({"yes":1, "no":0, "unknown":-1})
#dataframe['housing'] = dataframe['housing'].map({"yes":1, "no":0, "unknown":-1})
#dataframe['loan'] = dataframe['loan'].map({"yes":1, "no":0, "unknown":-1})
#dataframe['contact'] = dataframe['contact'].map({"cellular":1, "telephone":0})
#dataframe['month'] = dataframe['month'].map({'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12})
#dataframe['day_of_week'] = dataframe['day_of_week'].map({'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5})
#dataframe['poutcome'] = dataframe['poutcome'].map({'success':1, 'failure':0, 'nonexistent':-1})


# In[15]:


dataframe


# In[16]:


dataframe.shape


# # Correlation matrix
We have all our entries in numerical type, so we could explore more into the data to get more insights. For example, to check if there are correlations between variables. For that we compute the correlation coefficients and make a heat plot on Seaborn. The correlation matrix is shown in the following picture, with the scale of the correlation coefficients to the right. We see that there are some columns which are correlated. 
# In[17]:


#Now we check for correlations between the columns and we make a heatmap plot
sns.heatmap(dataframe.corr(),annot=False)
plt.rcParams['figure.figsize'] = (50.0, 50.0)    # ... and big plots
# We see no visible correlations among the columns, so we proceed with the analysis


# # Delete correlated columns from the database
The correlation coefficients computed before, say that there are some of the columns that they increase linearly with each other. Since our classification techniques take into account linear classifier, it means that if two variables are linearly classified, then we are introducing two different coefficients for the same column, leading to an overfitting; therefore, we should eliminate those correlated columns from the analysis as we don't want overfitting processes to occur. 
# In[18]:


corr_matrix = dataframe.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
to_drop = [column for column in dataframe.columns if any(upper[column]>0.75)]
dataframe=dataframe.drop(to_drop,axis=1)    

sns.heatmap(dataframe.corr(),annot=False)
plt.rcParams['figure.figsize'] = (50.0, 50.0)    # ... and big plots


# In[19]:


dataframe


# # Prepare the database for further analysis
Once deleted the correlated columns, the data is ready for the analysis. We will run several classification algorithms and check the predictions and the accuracy. For that we load the main libraries for Machine Learning, such as sklearn and the metrics to evaluate the results.
# In[20]:


# Database cleaning and preparation for the analysis
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# Evaluating the accuracy of the performed analysis
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

During any supervised machine learning algorithm, such as classification algorithms, we need to train our models with a training dataset and after, to test the model. For that, we separate our dataset into a training a test subsets. After we make the training subset, we scale them in the sense that we translate all the points in the corresponding hyper-space in such a way that the centroid becomes the origin of coordinates. Then we scale all the the points by a transformation that gives a standard deviation from the origin equals to 1. With such a process we homogeneize our sample for a better analysis and better learning for the algorithm.
# In[21]:


#Setting up the training and test variables. The test size is 20% of the total amount of data.
X = dataframe.iloc[:,:-1].values
Y = dataframe.iloc[:,-1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0, test_size = 0.3)

#We scale our training and test datasets to have zero mean and a standard deviation of 1.
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

In this section, we plot in polar coordinates the distribution of points around the centroid. The radial coordinate coordinate is the distance between the centroid and each point, and the angular coordinate is the angle between the centroid and the points in this hyperspace. The blue points correspond to the points classificed as having y=0 and the red points are those with y=1. If the points with y=0 and y=1 are well clustered, then the centroids will be somehow far away from each other. The distance between the centroids is around 383.87 units, and from both polar plots, we can see that the diameter of the clusters are larger than this distance, therefore, the clusters overlap, showing at the same time that the decision boundary is somehow tricky, making the decision even harder. 
# In[22]:


import scipy

def centroid_vector(t_array):
    return sum(t_array)/len(t_array)
    

def euclidean_distance(point,t_array):
    result =[]
    for el in t_array:
        result.append(scipy.spatial.distance.euclidean(point, el))
    return result

def angular_distribution(point,t_array):
    result = []
    for el in t_array:
        result.append((180.0/np.pi)*np.arccos(sum((a*b) for a, b in zip(point, el))/(np.linalg.norm(point)*np.linalg.norm(el))))
        
    return result


fraud = dataframe[dataframe["y"]==1]
notfraud = dataframe[dataframe["y"]==0]

print("The distance between the centroids is:", round(scipy.spatial.distance.euclidean(centroid_vector(fraud.values), centroid_vector(notfraud.values)),2))


theta_fraud = angular_distribution(centroid_vector(fraud.values),fraud.values)
theta_notfraud = angular_distribution(centroid_vector(notfraud.values),notfraud.values)

r_fraud = euclidean_distance(centroid_vector(fraud.values),fraud.values)
r_notfraud = euclidean_distance(centroid_vector(notfraud.values),notfraud.values)

fig = plt.figure()
ax1 = fig.add_subplot(121, polar=True)
c_fraud = ax1.scatter(theta_fraud, r_fraud, c='r', cmap='hsv', alpha=0.75)
ax1.set_thetamin(0)
ax1.set_thetamax(360)

ax2 = fig.add_subplot(122, polar=True)
c_notfraud = ax2.scatter(theta_notfraud, r_notfraud, c='b', cmap='hsv', alpha=0.75)
ax2.set_thetamin(0)
ax2.set_thetamax(360)

plt.rcParams['figure.figsize'] = (20.0, 20.0)    # ... and big plots
#ax3 = ax2.twinx()

#plt.show()


# # Logistic Regression
In this section we perform a Logistic Regression as a classification technique. We compute the confusion matrix and the accuracy of the model, giving a 90.75%. However, since the data is umbalanced, we should try other methods to see if we could improve the accuracy.
# In[23]:


from sklearn.linear_model import LogisticRegression

#Now it is time to start classifying our target data using the Logisitic Regression algorithm.
Log_Classifier = LogisticRegression()
Log_Classifier.fit(X_train,Y_train)
Y_pred_log = Log_Classifier.predict(X_test)

c_matrix_log = confusion_matrix(Y_test,Y_pred_log)
accgoal_log = accuracy_score(Y_test, Y_pred_log)
print(c_matrix_log)
print('The log determinant is:', np.log10(np.linalg.det(c_matrix_log)))
print("The accuracy goal is:",round(accgoal_log*100,2),"%")

sns.heatmap(c_matrix_log, annot=True, square=True, cmap = 'Blues_r')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# # kNN Algorithm
Since the database is not that large, we could use as well the kNN algorithm to perform a classification. We compute as well the confusion matrix and the accuracy of the model, giving a 90.39%. We obtain a similar result as with the Logistic Regression.
# In[24]:


from sklearn.neighbors import KNeighborsClassifier

### We will repeat the same analysis but now with the KNN algorithm

kNN = 11 #I will use this parameter and we see that the accuracy is better than the Logistic regression.
print('The k parameter is:', kNN)
classifier = KNeighborsClassifier(n_neighbors = kNN, p = 2, metric = 'euclidean' )
classifier.fit(X_train, Y_train)
Y_pred_knn = classifier.predict(X_test)

c_matrix_knn = confusion_matrix(Y_test, Y_pred_knn)
accgoaal_knn = accuracy_score(Y_test, Y_pred_knn)
print(c_matrix_knn)
print('The log determinant is:', np.log10(np.linalg.det(c_matrix_knn)))
print("The accuracy goal is:",round(accgoaal_knn*100,2),"%")


sns.heatmap(c_matrix_knn, annot=True, square=True, cmap = 'Reds_r')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# # Light Gradient Boosting algorithm
We perform as well a classification using the Light Gradient Boosting method with the parameters as defined below. We compute the confusion matrix and the accuracy of the model, giving a 89.58%, showing that this method under-performs when compared to kNN and Logistic Regression.
# In[25]:


import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# For this task we use a simple set of parameters to train the model. 
# We just want to create a baseline model, so we are not performing here cross validation or parameter tunning.

lgb_train = lgb.Dataset(X_train, Y_train, free_raw_data=False)
lgb_test = lgb.Dataset(X_test, Y_test, reference=lgb_train, free_raw_data=False)

parameters = {'num_leaves': 2**8,
              'learning_rate': 0.1,
              'is_unbalance': True,
              'min_split_gain': 0.1,
              'min_child_weight': 1,
              'reg_lambda': 1,
              'subsample': 1,
              'objective':'binary',
              #'device': 'gpu', # comment this line if you are not using GPU
              'task': 'train'
              }
num_rounds = 300


clf = lgb.train(parameters, lgb_train, num_boost_round=num_rounds)

Y_prob_gb = clf.predict(X_test)

Y_pred_gb = np.where(Y_prob_gb > 0.5, 1, 0)
cmatrix_gb = confusion_matrix(Y_test,Y_pred_gb)
accgoal_gb = accuracy_score(Y_test, Y_pred_gb)
print(cmatrix_gb)
print('The log determinant is:', np.log10(np.linalg.det(cmatrix_gb)))
print("The accuracy goal is:", round(accgoal_gb*100,2),'%')


plt.figure()
sns.heatmap(cmatrix_gb, annot=True, square=True, cmap = 'Purples_r')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.rcParams['figure.figsize'] = (5.0, 5.0)    # ... and big plots
print('-----------------------')


# # Random Forest algorithm
More into the Deep Learning field, we perform as well a classification using the Random Forest method with the parameters as defined below. We compute the confusion matrix and the accuracy of the model, giving a 90.53%, showing that this method performs similar to kNN and Logistic Regression.
# In[26]:


from sklearn.ensemble import RandomForestClassifier

classif_rmd = RandomForestClassifier(min_samples_leaf=150)
classif_rmd.fit(X_train, Y_train)
Y_pred_rmf = classif_rmd.predict(X_test)
accgoal_rmf = accuracy_score(Y_pred_rmf, Y_test)
cmatrix_rmf = confusion_matrix(Y_test, Y_pred_rmf)
print(cmatrix_rmf)
print('The log determinant is:', np.log10(np.linalg.det(cmatrix_rmf)))
print("The accuracy goal is:", round(accgoal_rmf*100,2),'%')


sns.heatmap(cmatrix_rmf, annot=True, square=True, cmap = 'Greens_r')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.rcParams['figure.figsize'] = (5.0, 5.0)    # ... and big plots


# # Empirical sigmoid function
So, this empirical sigmoid function came to my head when analyzing rotation curves of galaxies when I was doing my thesis. You could find the reference at Mon. Not. Roy. Astron. Soc. 468 (2017) no.1, 147-153. Stars and gas in galaxies rotate in form of a S-shaped curve with constant velocities in the external parts of the galaxy. From that analogy, I took this curve that can be used as well as a classifier because it ressembles a Sigmoid Function. 
# In[27]:


def empirical_sigmoid(x):
    return 0.5*(1.0 + x/(1.0+np.abs(x)))

xsig = np.linspace(-5,5,50)
ysig = empirical_sigmoid(xsig)

# plot visuals
f, ax = plt.subplots(figsize=(10, 5))
plt1 = plt.plot(xsig, ysig, color="orange")
plt.xticks(size=20,rotation=0)
plt.yticks(size=20,rotation=0)
ax.grid(True, which='both')
ax.axhline(y=0, color='r')
ax.axvline(x=0, color='r')

We perform as well a classification using this Empirical Sigmoid Function; we make a class called Empirical Classifier which works in general with any Sigmoid Function that may come to our heads. We compute as well the confusion matrix and the accuracy of the model, giving a 90.43%, showing that this method is as well as powerful as Random Forest, kNN and Logistic Regression.
# In[28]:


class Empirical_Classifier:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __empirical_class(self, z):
        return 0.5*(1.0 + z/(1.0+np.abs(z)))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__empirical_class(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            z = np.dot(X, self.theta)
            h = self.__empirical_class(z)
            loss = self.__loss(h, y)
                
            if(self.verbose ==True and i % 10000 == 0):
                print('floss: {loss} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__empirical_class(np.dot(X, self.theta))
    
    def predict(self, X):
        return self.predict_prob(X).round()

model = Empirical_Classifier(lr=0.1, num_iter=300000)
get_ipython().run_line_magic('time', 'model.fit(X_train, Y_train)')
Y_pred_emp = model.predict(X_test)
params = model.theta

accgoal_emp = accuracy_score(Y_pred_emp, Y_test)
c_matrix_emp = confusion_matrix(Y_test, Y_pred_emp)
print(c_matrix_emp)
print('The log determinant is:', np.log10(np.linalg.det(c_matrix_emp)))
print("The accuracy goal is:", round(accgoal_emp*100,2),'%')

sns.heatmap(c_matrix_emp, annot=True, square=True, cmap = 'Oranges_r')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.rcParams['figure.figsize'] = (8.0, 8.0)    # ... and big plots


# # Artificial Neural Networks
Inside the domains of the Deep Learning, we use this time a Neural Network algorithm with two hidden layers with RELU and SIGMOID activation functions between the input output layers respectively. We import the library Keras and TensorFlow to perform the analysis and compute as well the confusion matrix and the accuracy of the model, giving a 91.15%, showing that this method over-performs when compared to the previous results from the Empirical, Random Forest, kNN and Logistic Regression classifiers.  
# In[30]:


import keras
from keras.models import Sequential
from keras.layers import Dense# Initialising the ANN
"Activation functions: 'sigmoid', 'tanh', 'relu'"
classifier = Sequential()# Adding the input layer and the first hidden layer
classifier.add(Dense(units =15 , kernel_initializer = 'uniform', activation = 'relu', input_dim = 52))# Adding the second hidden layer
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])# Fitting the ANN to the Training set
classifier.fit(X_train, Y_train, batch_size = 32, epochs = 150)

# Predicting the Test set results
Y_pred_ann = classifier.predict(X_test)
Y_pred_ann = (Y_pred_ann > 0.5)
accgoal_ann = accuracy_score(Y_pred_ann, Y_test)

cmatrix_ann = confusion_matrix(Y_test, Y_pred_ann) # rows = truth, cols = prediction
print(cmatrix_ann)
print('The log determinant is:', np.log10(np.linalg.det(cmatrix_ann)))
print("The accuracy goal is:", round(accgoal_ann*100,2),'%')


sns.heatmap(cmatrix_ann, annot=True, square=True, cmap = 'Greys_r')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.rcParams['figure.figsize'] = (8.0, 8.0)    # ... and big plots


# # Comparison between the different classification algorithms
Here we compare the accuracy between the five different classifiers used in this analysis. In the order of better-to-worst accuracies we have: Artificial Neural Networks, Random Forest, Logistic Regression, kNN, Empirical Sigmoid and Light Gradient Boosting. 
# In[33]:


log_cols=["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)
log_entry = pd.DataFrame([['Logistic Regression',accgoal_log],['Artificial Neural Networks',accgoal_ann],['Empirical Sigmoid', accgoal_emp],['Light Gradient Boosting',accgoal_gb],['Random Forest', accgoal_rmf],['kNN',accgoaal_knn]], columns=log_cols)
        #metric_entry = pd.DataFrame([[precision,recall,f1_score,roc_auc]], columns=metrics_cols)
log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier vs. Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="deepskyblue")  
plt.rcParams['figure.figsize'] = (15.0, 5.0)    # ... and big plots
plt.show()


# # SMOTE: Generate synthetic samples
We saw at the introduction that our sample is not well balanced, as there are around 11.26% of the total cases subscribed a deposit. We will generate synthetic samples to balance the data and proceed with the estimation, in order to see if we could improve our predictions. For that, we use the SMOTE technique.

We take as a random state 27 and the ratio of subscribed and unsubscribed as 1, which means, we generate synthetic samples, well balanced this time. 
# In[35]:


from imblearn.over_sampling import SMOTE

sm = SMOTE(sampling_strategy='all', random_state=None, ratio=1.0)
X_train_smote, Y_train_smote = sm.fit_sample(X_train, Y_train)

sns.countplot(Y_train_smote)
plt.title('Balanced training data')
plt.rcParams['figure.figsize'] = (15.0, 5.0)
sns.set_palette(['blue', 'red'])
plt.show()


# # Logistic Regression SMOTE
We repeat the same analysis as before with the Logistic Regression, this time, using the synthetic sample generated by SMOTE. As we can see, this resampling technique doesn't improve the accuracy of the predictions.
# In[37]:


from sklearn.linear_model import LogisticRegression

#Now it is time to start classifying our target data using the Logisitic Regression algorithm.
Log_Classifier = LogisticRegression()
Log_Classifier.fit(X_train_smote,Y_train_smote)
Y_pred_log_smote = Log_Classifier.predict(X_test)

c_matrix_log_smote = confusion_matrix(Y_test,Y_pred_log_smote)
accgoal_log_smote = accuracy_score(Y_test, Y_pred_log_smote)
print(c_matrix_log_smote)
print('The log determinant is:', np.log10(np.linalg.det(c_matrix_log_smote)))
print("The accuracy goal is:",round(accgoal_log_smote*100,2),"%")

sns.heatmap(c_matrix_log_smote, annot=True, square=True, cmap = 'Blues_r')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# # kNN algorithm SMOTE
We repeat the same analysis as before with the kNN algorithm, this time, using the synthetic sample generated by SMOTE. As we can see, this resampling technique does improve the accuracy of the predictions.
# In[38]:


from sklearn.neighbors import KNeighborsClassifier

### We will repeat the same analysis but now with the KNN algorithm

kNN = 11 #I will use this parameter and we see that the accuracy is better than the Logistic regression.
print('The k parameter is:', kNN)
classifier_smote = KNeighborsClassifier(n_neighbors = kNN, p = 2, metric = 'euclidean' )
classifier_smote.fit(X_train_smote, Y_train_smote)
Y_pred_knn_smote = classifier.predict(X_test)
Y_pred_knn_smote = np.where(Y_pred_knn_smote>0.5,1.0,0.0)

c_matrix_knn_smote = confusion_matrix(Y_test, Y_pred_knn_smote)
accgoaal_knn_smote = accuracy_score(Y_test, Y_pred_knn_smote)
print(c_matrix_knn_smote)
print('The log determinant is:', np.log10(np.linalg.det(c_matrix_knn_smote)))
print("The accuracy goal is:",round(accgoaal_knn_smote*100,2),"%")


sns.heatmap(c_matrix_knn_smote, annot=True, square=True, cmap = 'Reds_r')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# # Light Grandient Boosting SMOTE
We repeat the same analysis as before with the Light Gradient Boosting algorithm, this time, using the synthetic sample generated by SMOTE. As we can see, this resampling technique underperforms, giving bad result and therefore not so good predictions.
# In[39]:


import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# For this task we use a simple set of parameters to train the model. 
# We just want to create a baseline model, so we are not performing here cross validation or parameter tunning.

lgb_train_smote = lgb.Dataset(X_train_smote, Y_train_smote, free_raw_data=False)
lgb_test_smote = lgb.Dataset(X_test, Y_test, reference=lgb_train_smote, free_raw_data=False)

parameters = {'num_leaves': 2**8,
              'learning_rate': 0.1,
              'is_unbalance': True,
              'min_split_gain': 0.1,
              'min_child_weight': 1,
              'reg_lambda': 1,
              'subsample': 1,
              'objective':'binary',
              #'device': 'gpu', # comment this line if you are not using GPU
              'task': 'train'
              }
num_rounds = 300


clf_smote = lgb.train(parameters, lgb_train_smote, num_boost_round=num_rounds)

Y_prob_gb_smote = clf_smote.predict(X_test)

Y_pred_gb_smote = np.where(Y_prob_gb_smote > 0.5, 1, 0)
cmatrix_gb_smote = confusion_matrix(Y_test,Y_pred_gb_smote)
accgoal_gb_smote = accuracy_score(Y_test, Y_pred_gb_smote)
print(cmatrix_gb_smote)
print('The log determinant is:', np.log10(np.linalg.det(cmatrix_gb_smote)))
print("The accuracy goal is:", round(accgoal_gb_smote*100,2),'%')


plt.figure()
sns.heatmap(cmatrix_gb_smote, annot=True, square=True, cmap = 'Purples_r')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.rcParams['figure.figsize'] = (5.0, 5.0)    # ... and big plots
print('-----------------------')


# # Random Forest SMOTE
We repeat the same analysis as before with the Random Forest algorithm, this time, using the synthetic sample generated by SMOTE. As we can see, this resampling technique doesn't improve the accuracy of the predictions.
# In[40]:


from sklearn.ensemble import RandomForestClassifier

classif_rmd_smote = RandomForestClassifier(min_samples_leaf=150)
classif_rmd_smote.fit(X_train_smote, Y_train_smote)
Y_pred_rmf_smote = classif_rmd_smote.predict(X_test)
accgoal_rmf_smote = accuracy_score(Y_pred_rmf_smote, Y_test)
cmatrix_rmf_smote = confusion_matrix(Y_test, Y_pred_rmf_smote)
print(cmatrix_rmf_smote)
print('The log determinant is:', np.log10(np.linalg.det(cmatrix_rmf_smote)))
print("The accuracy goal is:", round(accgoal_rmf_smote*100,2),'%')


sns.heatmap(cmatrix_rmf_smote, annot=True, square=True, cmap = 'Greens_r')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.rcParams['figure.figsize'] = (5.0, 5.0)    # ... and big plots


# # Empirical Sigmoid SMOTE
We repeat the same analysis as before with the Empirical Classifier, this time, using the synthetic sample generated by SMOTE. As we can see, this resampling technique doesn't improve the accuracy of the predictions.
# In[41]:


class Empirical_Classifier:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __empirical_class(self, z):
        return 0.5*(1.0 + z/(1.0+np.abs(z)))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__empirical_class(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            z = np.dot(X, self.theta)
            h = self.__empirical_class(z)
            loss = self.__loss(h, y)
                
            if(self.verbose ==True and i % 10000 == 0):
                print('floss: {loss} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__empirical_class(np.dot(X, self.theta))
    
    def predict(self, X):
        return self.predict_prob(X).round()

model = Empirical_Classifier(lr=0.1, num_iter=300000)
get_ipython().run_line_magic('time', 'model.fit(X_train_smote, Y_train_smote)')
Y_pred_emp_smote = model.predict(X_test)
params = model.theta

accgoal_emp_smote = accuracy_score(Y_pred_emp_smote, Y_test)
c_matrix_emp_smote = confusion_matrix(Y_test, Y_pred_emp_smote)
print(c_matrix_emp_smote)
print('The log determinant is:', np.log10(np.linalg.det(c_matrix_emp_smote)))
print("The accuracy goal is:", round(accgoal_emp_smote*100,2),'%')

sns.heatmap(c_matrix_emp_smote, annot=True, square=True, cmap = 'Oranges_r')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.rcParams['figure.figsize'] = (8.0, 8.0)    # ... and big plots


# # Artificial Neural Networks SMOTE
We repeat the same analysis as before with the Artificial Neural Networks, this time, using the synthetic sample generated by SMOTE. As we can see, this resampling technique doesn't improve the accuracy of the predictions.
# In[44]:


import keras
from keras.models import Sequential
from keras.layers import Dense# Initialising the ANN
"Activation functions: 'sigmoid', 'tanh', 'relu'"
classifier = Sequential()# Adding the input layer and the first hidden layer
classifier.add(Dense(units =15 , kernel_initializer = 'uniform', activation = 'relu', input_dim = 52))# Adding the second hidden layer
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])# Fitting the ANN to the Training set
classifier.fit(X_train_smote, Y_train_smote, batch_size = 32, epochs = 100)

# Predicting the Test set results
Y_pred_ann_smote = classifier.predict(X_test)
Y_pred_ann_smote = (Y_pred_ann_smote > 0.5)
accgoal_ann_smote = accuracy_score(Y_pred_ann_smote, Y_test)

cmatrix_ann_smote = confusion_matrix(Y_test, Y_pred_ann_smote) # rows = truth, cols = prediction
print(cmatrix_ann_smote)
print('The log determinant is:', np.log10(np.linalg.det(cmatrix_ann_smote)))
print("The accuracy goal is:", round(accgoal_ann_smote*100,2),'%')


sns.heatmap(cmatrix_ann_smote, annot=True, square=True, cmap = 'Greys_r')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.rcParams['figure.figsize'] = (8.0, 8.0)    # ... and big plots

Here, as previously done, we compare the accuracy between the five different classifiers used in this analysis with the SMOTE. In the order of better-to-worst accuracies we have: kNN, Artificial Neural Networks, Logistic Regression, Empirical Sigmoid, Random Forest, and Light Gradient Boosting. 
# In[47]:


log_cols=["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)
log_entry = pd.DataFrame([['kNN',accgoaal_knn_smote],['Logistic Regression',accgoal_log_smote],['Artificial Neural Networks',accgoal_ann_smote],['Empirical Sigmoid', accgoal_emp_smote], ['Light Gradient Boosting',accgoal_gb_smote], ['Random Forest', accgoal_rmf_smote]], columns=log_cols)
        #metric_entry = pd.DataFrame([[precision,recall,f1_score,roc_auc]], columns=metrics_cols)
log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier vs. Accuracy using Synthetic Minority Over-sampling TEchnique')
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="deepskyblue")  
plt.rcParams['figure.figsize'] = (15.0, 5.0)    # ... and big plots
plt.show()

One could see here that some algorithms gain more accuracy when doing SMOTE than before, like for example, kNN which improves much better. I will do a general comparative bar plot resume for the accuracy of each algorithm. 
# In[58]:


log_cols=["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)
log_entry = pd.DataFrame([['Logistic Regression',accgoal_log],['Artificial Neural Networks',accgoal_ann],['kNN_SMOTE',accgoaal_knn_smote],['Logistic Regression_SMOTE',accgoal_log_smote],['Artificial Neural Networks_SMOTE',accgoal_ann_smote],['Empirical Sigmoide', accgoal_emp],['Empirical Sigmoide_SMOTE', accgoal_emp_smote], ['Light Gradient Boosting',accgoal_gb], ['Light Gradient Boosting_SMOTE',accgoal_gb_smote], ['Random Forest', accgoal_rmf],['Random Forest_SMOTE', accgoal_rmf_smote],['kNN',accgoaal_knn]], columns=log_cols)
        #metric_entry = pd.DataFrame([[precision,recall,f1_score,roc_auc]], columns=metrics_cols)
log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier vs. Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="deepskyblue")  
plt.rcParams['figure.figsize'] = (15.0, 5.0)    # ... and big plots
plt.show()

