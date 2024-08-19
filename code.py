########### SOLAR POWER PREDICTION MODEL P-182

'''
CRISP-ML(Q) process model describes six phases:
# - Business and Data Understanding
# - Data Preparation
# - Model Building
# - Model Evaluation and Hyperparameter Tuning
# - Model Deployment
# - Monitoring and Maintenance

Client:
One of the leading companies in solar power generation.

Business Problem:
Manual inspection limitations lead to undetected defects, decreasing energy production and systems.

Business Objective:
Maximise energy production and system reliability through efficient defect detection.

Business Constraint:
Minimize resource utilization.
Maximizing energy production and system reliability.

Success criteria:
Business success criteria:
Increase in energy production efficiently by at least 10% and improvement in system reliability with a 20% reduction in downtime.

Machine Learning success criteria:Achieve an accuracy of at least 95%.

Economic success criteria:
Reduction in maintenance costs by 15% and Increase in return on investment (ROI) by 20% through improved efficiency.
    

Data Description:
    
GPVS-Faults: Experimental Data for fault scenarios in grid-connected PV systems under MPPT and IPPT modes

Overview:
The Grid-connected PV System Faults (GPVS-Faults) data are collected from lab experiments of faults in a PV microgrid system. 
Experiment scenario, including photovoltaic array faults; inverter faults; grid anomalies; feedback sensor fault; and MPPT controller faults of various severity. 
GPVS-Faults data can be used to design/ validate/ compare various algorithms of fault detection/ diagnosis/ classification for PV system protection and reactive maintenance.

Description: 
The faults were introduced manually halfway during the experiments. 
The high-frequency measurements are noisy; with disturbances and variations of temperature and insolation during and between the experiments; MPPT/IPPT modes have adverse effects on the detection of low-magnitude faults. 
After critical faults, the operation is interrupted and the system may shut-down; the challenge is to detect the faults before a total failure.


Data Collection: 
        Dimension: 12294 rows and 7 columns
        
Time: Time in seconds, average sampling T_s=9.9989 μs.
Ipv: PV array current measurement. 
Vpv: PV array voltage measurement.
Vdc: DC voltage measurement. 
ia, ib, ic: 3-Phase current measurements. 
va, vb, vc: 3-Phase voltage measurements.
Iabc: Current magnitude.  
If: Current frequency.
Vabc: Voltage magnitude.  
Vf: Voltage frequency.	
Defecitive/Non Defective: {0: Non-Defective, 1: Defective}  
 
'''


# Importing all required libraries, modules
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import Pipeline

import pickle

import warnings 
warnings.filterwarnings('ignore')

import joblib


# Load the datasets into Python dataframe
data = pd.read_csv(r"solar_data.csv")
print(data.head())

#data = pd.read_excel(r"gpvs_faults.xlsx")
#print(data.head())

# Remove trailing space from column name
# data.rename(columns={'Defective/Non Defective ': 'Defective/Non Defective'}, inplace=True)

# Database Connection
from sqlalchemy import create_engine,text

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="root",# user
                               pw="root", # passwrd
                               db="dsproject")) #database

# Upload the Table into Database
data.to_sql('solar_data',con = engine, if_exists = 'replace', chunksize = 1000, index = False)


# Create a connection
connection = engine.connect()

# Read the Table (data) from MySQL database
sql = 'SELECT * FROM solar_data;'
# datadf = pd.read_sql_query(sql, con = engine)
data=pd.read_sql_query(text(sql),engine.connect())

data.head()

print(data)

data.columns

data.head(5)

data.tail(5)

data.describe()

data.info()

data['Defective / Non Defective']


# Dropping data 
data = data.drop(columns = 'Time', axis = 1)
data

###  Data checking 

# Check for Missing values
data['Defective / Non Defective'].isnull().sum() 


## Check Duplicates 
data.duplicated().sum()

duplicates = data[data.duplicated()]
duplicates

### Check  Balanced or Un-Balanced

# Set the column to be plotted on the x-axis
x = 'Defective / Non Defective'

# Create the count plot
sns.countplot(x = x, data = data)

# tile for the plot
plt.title('Checking whether the data is Balanced or Un-Balanced')

# Show the plot
plt.show()



#### Data Preprocessing

## First Moment Business Decision / Measures of Central Tendency

# mean
mean = data.mean()
print(mean)

# Now for other columns
columns_to_process = ['Time', 'Ipv', 'Vpv', 'Vdc', 'ia', 'ib', 'ic', 'va', 'vb', 'vc', 'Iabc', '_If', 'Vabc', 'Vf']

for column in columns_to_process:
    mean = data[column].mean()
    print("Mean of", column + ":", mean)

# median
median = data.median()
print(median)

# Now for other columns
columns_to_process = ['Time', 'Ipv', 'Vpv', 'Vdc', 'ia', 'ib', 'ic', 'va', 'vb', 'vc', 'Iabc', '_If', 'Vabc', 'Vf']

for column in columns_to_process:
    median = data[column].median()
    print("Median of", column + ":", median)

# mode
mode = data.mode()
print(mode)


mode = data["Defective / Non Defective"].mode()
print(mode)


## Second Moment Business Decision / Measures of Dispersion

# Standard Deviation of Solar data
data_stddev = data.std()

print("Standard Deviation of Solar_data:", data_stddev)

# Now for other columns
columns_to_process = ['Time', 'Ipv', 'Vpv', 'Vdc', 'ia', 'ib', 'ic', 'va', 'vb', 'vc', 'Iabc', '_If', 'Vabc', 'Vf']

for column in columns_to_process:
    stddev = data[column].std()
    print("Standard Deviation of", column + ":", stddev)

# Range of Solar data
data_range = data.max() - data.min()
print("Range of solar_data:", data_range)

# Now for other columns
# columns_to_process = ['Time', 'Ipv', 'Vpv', 'Vdc', 'ia', 'ib', 'ic', 'va', 'vb', 'vc', 'Iabc', '_If', 'Vabc', 'Vf', 'defective_or_non_defective']

for column in columns_to_process:
    range_column = data[column].max() - data[column].min()
    print("Range of solar_data", column + ":", range_column)

# Variance of Solar data
variance = data.var()
print("Variance of solar_data:", variance)

# Now for other columns
# columns_to_process = ['Time', 'Ipv', 'Vpv', 'Vdc', 'ia', 'ib', 'ic', 'va', 'vb', 'vc', 'Iabc', '_If', 'Vabc', 'Vf', 'defective_or_non_defective']

for column in columns_to_process:
    variance_column = data[column].var()
    print("Variance of", column + ":", variance_column)


## Third Moment Business Decision / Skewness

skewness = data.skew()
print(skewness)

# Now for other columns
# columns_to_process = ['Time', 'Ipv', 'Vpv', 'Vdc', 'ia', 'ib', 'ic', 'va', 'vb', 'vc', 'Iabc', '_If', 'Vabc', 'Vf', 'defective_or_non_defective']

for column in columns_to_process:
    skewness_column = data[column].skew()
    print("Skewness of", column + ":", skewness_column)
    
    
## Fourth Moment Business Decision / Kurtosis

kurtosis = data.kurtosis()
print(kurtosis)


# Now for other columns
# columns_to_process = ['Time', 'Ipv', 'Vpv', 'Vdc', 'ia', 'ib', 'ic', 'va', 'vb', 'vc', 'Iabc', '_If', 'Vabc', 'Vf', 'defective_or_non_defective']

for column in columns_to_process:
    kurtosis_column = data[column].kurtosis()
    print("Kurtosis of", column + ":", kurtosis_column)
    

## Graphical Representation

plt.figure(figsize=(10, 6))

# Plotting the Time in seconds
plt.hist(data['Time'], bins=20)
plt.title('Time ')
plt.tight_layout()
plt.show()

### Checking outlier_Boxplot
sns.boxplot(data.Vabc); plt.title('Vabc'); plt.show()
sns.boxplot(data.Ipv); plt.title('Ipc'); plt.show()
sns.boxplot(data.Vdc); plt.title('Vdc'); plt.show()
sns.boxplot(data.ia); plt.title('ia'); plt.show()
sns.boxplot(data.ib); plt.title('ib'); plt.show()
sns.boxplot(data.ic); plt.title('ic'); plt.show()
sns.boxplot(data.va); plt.title('va'); plt.show()
sns.boxplot(data.vb); plt.title('vb'); plt.show()
sns.boxplot(data.vc); plt.title('vc'); plt.show()
sns.boxplot(data.Iabc); plt.title('Iabc'); plt.show()
sns.boxplot(data._If); plt.title('_If'); plt.show()
sns.boxplot(data.Vf); plt.title('Vf'); plt.show()



## Histogram for individual coloums

# Define the columns and their descriptions
columns_info = {
    'Time': 'Time in seconds, average sampling T_s=9.9989 μs.',
    'Ipv': 'PV array current measurement.',
    'Vpv': 'PV array voltage measurement.',
    'Vdc': 'DC voltage measurement.',
    'ia': 'Phase A current measurement.',
    'ib': 'Phase B current measurement.',
    'ic': 'Phase C current measurement.',
    'va': 'Phase A voltage measurement.',
    'vb': 'Phase B voltage measurement.',
    'vc': 'Phase C voltage measurement.',
    'Iabc': 'Current magnitude.',
    '_If': 'Current frequency.',
    'Vabc': 'Voltage magnitude.',
    'Vf': 'Voltage frequency.',
}

# Plot histograms for each specified column
for column in columns_to_process:
    plt.figure(figsize=(10, 6))
    plt.hist(data[column], bins=20)
    plt.title(columns_info[column])
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    
## or

plt.hist(data.Iabc); plt.title('Iabc'); plt.show()
plt.hist(data.Vabc); plt.title('Vabc'); plt.show()
plt.hist(data.Ipv); plt.title('Ipv'); plt.show()
plt.hist(data.Vpv); plt.title('Vpv'); plt.show()
plt.hist(data.ia); plt.title('ia'); plt.show()
plt.hist(data.ib); plt.title('ib'); plt.show()
plt.hist(data.ic); plt.title('ic'); plt.show()
plt.hist(data.va); plt.title('va'); plt.show()
plt.hist(data.vb); plt.title('vb'); plt.show()
plt.hist(data.vc); plt.title('vc'); plt.show()
plt.hist(data._If); plt.title('_If'); plt.show()
plt.hist(data.Vf); plt.title('Vf'); plt.show()


## Density Plot for all the variables in the datset

sns.distplot(data.Iabc); plt.title('Iabc'); plt.show()
sns.distplot(data.Vabc); plt.title('Vabc'); plt.show()
sns.distplot(data.Ipv); plt.title('Ipv'); plt.show()
sns.distplot(data.Vpv); plt.title('Vpv'); plt.show()
sns.distplot(data.ia); plt.title('ia'); plt.show()
sns.distplot(data.ib); plt.title('ib'); plt.show()
sns.distplot(data.ic); plt.title('ic'); plt.show()
sns.distplot(data.va); plt.title('va'); plt.show()
sns.distplot(data.vb); plt.title('vb'); plt.show()
sns.distplot(data.vc); plt.title('vc'); plt.show()
sns.distplot(data._If); plt.title('_If'); plt.show()
sns.distplot(data.Vf); plt.title('Vf'); plt.show()

    
    
# Correlation coefficient
data.corr()

# Create a heatmap
plt.figure(figsize = (15,10))
sns.heatmap(data.corr(), annot = True, cmap = 'coolwarm')


# Creating scatter plot for all columns 
sns.pairplot(data, height = 1.5)


## Auto EDA
# ## Automated Libraries

# AutoEDA
import sweetviz
my_report = sweetviz.analyze([data, "data"])

my_report.show_html('Report.html')


# D-Tale
########

# pip install dtale   # In case of any error then please install werkzeug appropriate version (pip install werkzeug==2.0.3)

# pip install dtale --no-deps

import dtale # for jupyter "! pip install dtale

d = dtale.show(data)
d.open_browser()

## 
#Install PyQt5 if you get this warning message - "UserWarning:Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure."
#pip install PyQt5
import PyQt5

data.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 


### Pandas Profiling
from pandas_profiling import ProfileReport

data = pd.read_csv('solar_power.csv') 

# Generate a profile report
profile = ProfileReport(data)

# Generate a profile report without specifying a configuration file
profile = ProfileReport(data, config_file="")

# Save the report to an HTML file
# Replace 'profile_report.html' with the desired file name
profile.to_file("profile_report.html")


####  Target variable categories
data['Defective / Non Defective '].unique()

# Data split into Input and Output
x = data.iloc[:, :13] # Predictors 
print(x)

y = data.iloc[:, 13:] # Predictors 
print(y)


numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
numeric_features

categorical_features = data.select_dtypes(include=['object']).columns
categorical_features

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from feature_engine.outliers import Winsorizer

impute = SimpleImputer(strategy='median')

data_columns = ['Ipv', 'Vpv', 'Vdc', 'ia', 'ib', 'ic', 'va', 'vb', 'vc', 'Iabc', '_If', 'Vabc', 'Vf']
X = data[data_columns]
y = data['Defective / Non Defective']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # standardize features
    ('clf', LogisticRegression())  # Logistic Regression model
])

imputation_data = ('Ipv', 'Vpv', 'Vdc', 'ia', 'ib', 'ic', 'va', 'vb', 'vc', 'Iabc', '_If', 'Vabc','Vf')

imputer = SimpleImputer(strategy='median')
imputer.fit(data)

imputer = SimpleImputer(strategy='median')
imputer.fit(X_train)
X_train = imputer.transform(X_train)

data['Defective / Non Defective']


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import sklearn.metrics as skmet
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report

# Checking unique categories/ classes in the Output 
data['Defective / Non Defective'].unique()
data['Defective / Non Defective'].value_counts()


# Defining Input and Output Variable 
X = data.drop('Defective / Non Defective', axis=1)
y = data['Defective / Non Defective']

# Splitting data into Training & Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

X_train.shape

X_test.shape

# Train the decision tree classifier
DT = DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion = 'entropy' , max_depth = 10)
DT.fit(X_train, y_train)


# Evaluate on Train Data

#Predict on the Train Data
DT_pred_Train = DT.predict(X_train)

# Accuracy on Train Data
DT_Train_Accuracy = accuracy_score(y_train, DT_pred_Train)
print("DT_Train_Accuracy:", DT_Train_Accuracy)


#Confusion Matrix on Train Data using Decision Tree
pd.crosstab(y_train, DT_pred_Train, rownames = ['Actual'], colnames= ['Predictions'])

# Precison, Recall, F1-Score, Support Params
print(classification_report(y_train, DT_pred_Train))


# Evalute on Test data

# Predict on the test set
DT_pred_Test = DT.predict(X_test)

# Accuracy on Test Data
Test_DT_Accuracy = accuracy_score(y_test, DT_pred_Test)
print("DT_Test_Accuracy:", Test_DT_Accuracy)


#Confusion Matrix on Test Data using Decision Tree
pd.crosstab(y_test, DT_pred_Test, rownames = ['Actual'], colnames= ['Predictions']) 


# Precison, Recall, F1-Score, Support Params
print(classification_report(y_test, DT_pred_Test))



### Decision Tree with GridSearchCV
# Decision tree model
dtree_model = DecisionTreeClassifier()

# Define Hyperparameters grid for GridSearchCV
param_grid = { 'criterion':['gini','entropy'], 'max_depth': np.arange(3, 21)}

# Initialise GridSearchCV
dtree_gscv = GridSearchCV(dtree_model, param_grid, cv = 5, scoring = 'accuracy', return_train_score = False, verbose = 1)

# Fit GridSearchCV to the Data
dt_gsc = dtree_gscv.fit(X, y)

# Print the best parameters
print("Best parameters: ", dt_gsc.best_params_)
print('Best score: ', dt_gsc.best_score_)
dt_accuracy = dt_gsc.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(dt_accuracy) )



# Train the decision tree classifier
DT = DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion = 'gini' , max_depth = 12)
DT.fit(X_train, y_train)

# Evaluate on Train Data
#Predict on the Train Data
DT_pred_Train = DT.predict(X_train)

# Accuracy on Train Data
DT_Train_Accuracy = accuracy_score(y_train, DT_pred_Train)
print("DT_Train_Accuracy:", DT_Train_Accuracy)

# Evalute on Test data
# Predict on the test set
DT_pred_Test = DT.predict(X_test)

# Accuracy on Test Data
Test_DT_Accuracy = accuracy_score(y_test, DT_pred_Test)
print("DT_Test_Accuracy:", Test_DT_Accuracy)


# Precison, Recall, F1-Score, Support Params
print(classification_report(y_train, DT_pred_Train))


# Precison, Recall, F1-Score, Support Params
print(classification_report(y_test, DT_pred_Test))



### RANDOM FOREST 

rfc = RandomForestClassifier(n_estimators = 100, max_depth = 15, random_state = 42)
rfc.fit(X_train, y_train)


# Evaluate on Train Data

#Predict on the Train Data
rfc_pred_train = rfc.predict(X_train)

# Calculate the accuracy of the model on the training set
rfc_train_accuracy = accuracy_score(y_train, rfc_pred_train)

# Accuracy on Train Data
print("RandomForest_Train_Accuracy is" , rfc_train_accuracy)


#Confusion Matrix on Train Data using Random Forest Classifier
pd.crosstab(y_train, rfc_pred_train, rownames = ['Actual'], colnames= ['Predictions']) 

# Precison, Recall, F1-Score, Support Params
print(classification_report(y_train, rfc_pred_train))



# Evaluate on Test Data

#Predict on the Test Data
rfc_pred_test = rfc.predict(X_test)

# Calculate the accuracy of the model on the testing set
rfc_test_accuracy = accuracy_score(y_test, rfc_pred_test)

# Accuracy on test Data
print("Random_Forest_Test_Accuracy is" , rfc_test_accuracy)



#Confusion Matrix on Test Data using Random Forest Classifier
pd.crosstab(y_test, rfc_pred_test, rownames = ['Actual'], colnames= ['Predictions']) 


# Precison, Recall, F1-Score, Support Params
print(classification_report(y_test, rfc_pred_test))



# Random Forest model
rf_model = RandomForestClassifier()

# Define Hyperparameters grid for GridSearchCV
params_rf = param_grid = {'n_estimators': [100, 200, 300, 400, 500], 'max_depth': np.arange(3, 15)}

# Initialise GridSearchCV
rf_gscv = GridSearchCV(rf_model, params_rf, cv = 5, scoring = 'accuracy', return_train_score = False, verbose = 1)

# Fit GridSearchCV to the Data
rf_gsc = rf_gscv.fit(X, y)

# Print the best parameters
print("Best parameters: ", rf_gsc.best_params_)
print('Best score: ', rf_gsc.best_score_)
rf_accuracy = rf_gsc.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(rf_accuracy) )


### Cross Validation on RandomForest

from sklearn.ensemble import RandomForestClassifier

# create the model
model = RandomForestClassifier()

from sklearn.model_selection import cross_val_score
# perform cross-validation
scores = cross_val_score(model, x, y, cv = 10)


# calculate the mean and standard deviation of the scores
mean_score = np.mean(scores)
std_dev = np.std(scores)

plt.plot(scores, label='Accuracy')
plt.fill_between(range(len(scores)), scores-std_dev, scores+std_dev, color = 'gray', alpha = 0.2)
plt.axhline(y = mean_score, color = 'red', linestyle = '--', label = 'Mean accuracy')
plt.xlabel('CV iteration')
plt.ylabel('Accuracy')
plt.title('Cross-validation results')
plt.legend()
plt.show()

# print the average accuracy
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#####  Logistic Regression

# Create and fit a logistic regression model
# Create list of different solvers
solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']


# loop through solvers and fit logistic regression model
for solver in solvers:
    logreg = LogisticRegression(solver=solver)
    logreg.fit(X_train, y_train)
    
# make predictions on train and test data
y_train_pred = logreg.predict(X_train)
y_test_pred = logreg.predict(X_test)

# calculate train and test accuracy
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)


# Precison, Recall, F1-Score, Support Params
print(classification_report(y_train, y_train_pred))
print(classification_report(y_test, y_test_pred))


print(f"Solver: {solver}")
print(f"Train_Accuracy: {train_acc:.3f}")
print(f"Test_Accuracy: {test_acc:.3f}")
print("")



# Define the set of solvers to be tested
solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

# Define the parameter grid for GridSearchCV
param_grid = {'solver': solvers, 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# Create an instance of Logistic Regression
log_reg = LogisticRegression()

# Create an instance of GridSearchCV
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV to the data
reg_gsc = grid_search.fit(X, y)

# Get the best parameters and the best score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
reg_accuracy = reg_gsc.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(reg_accuracy) )



#### K Nearest Neighbors Classifier

# Create a Knn Classifier
knn = KNeighborsClassifier(n_neighbors = 5)

# Train the Knn Model
knn.fit(X_train, y_train)

# Evaluate the Model with Train Data
# Predict on Train Data
knn_pred_train = knn.predict(X_train)  
knn_pred_train


# Confusion Matrix on Train Data
pd.crosstab(y_train, knn_pred_train, rownames = ['Actual'], colnames = ['Predictions']) 


# Accuracy Measure on Train Data
print(skmet.accuracy_score(y_train, knn_pred_train))


# Precison, Recall, F1-Score, Support Params
print(classification_report(y_train, knn_pred_train))


# Predict on Test Data
knn_pred_test = knn.predict(X_test)
knn_pred_test

# Confusion Matrix on Test Data
pd.crosstab(y_test, knn_pred_test, rownames = ['Actual'], colnames = ['Predictions'])

print(skmet.accuracy_score(y_test, knn_pred_test)) 


# Precison, Recall, F1-Score, Support Params
print(classification_report(y_test, knn_pred_test))


cm = skmet.confusion_matrix(y_test, knn_pred_test)

cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=['Faulty', 'Not Faulty'])
cmplot.plot()
cmplot.ax_.set(title = 'Solar Panel Fault Detection - kNN- Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')


# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(3, 50, 2):
    neigh = KNeighborsClassifier(n_neighbors = i)
    neigh.fit(X_train, y_train)
    train_acc = np.mean(neigh.predict(X_train) == y_train)
    test_acc = np.mean(neigh.predict(X_test) == y_test)
    diff = train_acc - test_acc
    acc.append([diff, train_acc, test_acc])
acc


# Plotting the data accuracies in a single plot
# Train Data Accuracy Plot
plt.plot(np.arange(3, 50, 2), [i[1] for i in acc], "ro-")
# Test Data Accuracy Plot 
plt.plot(np.arange(3, 50, 2), [i[2] for i in acc], "bo-")


### k-Nearest Neighbors (k-NN) with GridSearchCV
Knn = KNeighborsClassifier()

#Define Hyperparameters grid for grid search 
params_Knn = {'n_neighbors': np.arange(3, 50)}

# Initialize GridSearchCV
Knn_gs = GridSearchCV(knn, params_Knn, cv = 5, scoring = 'accuracy', return_train_score = False, verbose = 1 )

# Fit GridSearchCV to the Data
Knn_new = Knn_gs.fit(X_train, y_train)

#Get the best Hyperparameters
print("Best Parameters:", Knn_new.best_params_)
print('Best score: ', Knn_new.best_score_)
Knn_accuracy = Knn_new.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(Knn_accuracy) )


### SVM

from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Build the SVM model
svm_model = SVC(kernel ='linear', C = 1)
svm_model.fit(X_train, y_train)
svm_model.fit(X_test, y_test)


# Make predictions on the test set
y_pred_test = svm_model.predict(X_test)

# Make predictions on the train set
y_pred_train = svm_model.predict(X_train)


# Evaluation Metrics on test set
acc = accuracy_score(y_test, y_pred_test)

# Print the evaluation metrics
print("Accuracy:", acc)


# Precison, Recall, F1-Score, Support Params
print(classification_report(y_test, y_pred_test))



# Evaluation Metrics on train set
acc = accuracy_score(y_train, y_pred_train)

# Print the evaluation metrics
print("Accuracy:", acc)


# Precison, Recall, F1-Score, Support Params
print(classification_report(y_train, y_pred_train))



## Save the Best Model with Pickel library

# Save the model
pickle.dump(rfc, open('rfc.pkl', 'wb'))

import os
os.getcwd()

# Load data from a Pickle file
with open('rfc.pkl', 'wb') as f:
    loaded_data = pickle.load(f)
print(loaded_data)



# Save data to a Joblib file
import joblib
joblib.dump(rfc, 'rfc.joblib')

# Load data from a Joblib file

from sklearn.externals import joblib

loaded_data = joblib.load('rfc.joblib')
print(loaded_data)



