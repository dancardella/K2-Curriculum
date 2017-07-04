
# coding: utf-8

# In[75]:

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, scale
from sklearn.model_selection import cross_val_score
from sklearn import tree

import pydotplus
from IPython.display import Image

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import graphviz
import seaborn as sns
import sqlite3

from __future__ import division
from datetime import datetime

# No Real Reason For the Below
# %matplotlib inline


# Reading In Data
data= pd.read_csv("EEG data.csv")
col_headings =["Subject ID","Video ID","Attention","Mediation","Raw","Delta","Theta","Alpha1","Alpha2",
                    "Beta1","Beta2","Gamma1","Gamma2","expectconfusion","userconfusion"]
data.columns = col_headings 
#data.tail()


# In[3]:

#subject ID, age, ethnicity, gender
eeg_meta = [0,25,"Han Chinese","M",
1,24,"Han Chinese","M",
2,31,"English","M",
3,28,"Han Chinese","F",
4,24,"Bengali","M",
5,24,"Han Chinese","M",
6,24,"Han Chinese","M",
7,25,"Han Chinese","M",
8,25,"Han Chinese","M",
9,24,"Han Chinese","F"]


# In[ ]:

#Column 1: Subject ID
#Column 2: Video ID
#Column 3: Attention (Proprietary measure of mental focus)
#Column 4: Mediation (Proprietary measure of calmness)
#Column 5: Raw (Raw EEG signal)
#Column 6: Delta (1-3 Hz of power spectrum)
#Column 7: Theta (4-7 Hz of power spectrum)
#Column 8: Alpha 1 (Lower 8-11 Hz of power spectrum)
#Column 9: Alpha 2 (Higher 8-11 Hz of power spectrum)
#Column 10: Beta 1 (Lower 12-29 Hz of power spectrum)
#Column 11: Beta 2 (Higher 12-29 Hz of power spectrum)
#Column 12: Gamma 1 (Lower 30-100 Hz of power spectrum)
#Column 13: Gamma 2 (Higher 30-100 Hz of power spectrum)
#Column 14: predefined label (whether the subject is expected to be confused)
#Column 15: user-defined label (whether the subject is actually confused)


# ## EDA 

# In[5]:

data.info()


# In[87]:

data.iloc[:,2:].describe()


# In[86]:

data.iloc[:,2:].corr()


# In[107]:

correlation =data.iloc[:,2:].corr()
plt.figure(figsize = (14,14))
sns.heatmap(correlation, vmax= 1, square= True, annot= True, cmap="viridis")


# In[6]:

# KEY: 0=F, 1=M
data['Gender']= None
for idx, row in data.iterrows():
    if row['Subject ID'] == 3 or row['Subject ID'] == 9:
        data.loc[idx, 'Gender'] = 0
    else:
        data.loc[idx, 'Gender'] = 1


# In[7]:

# KEY: 0=CHINESE, 1=ENGLISH, 2= INDIAN
data['Ethnicity']= None
for idx, row in data.iterrows():
    if row['Subject ID'] == 2:
        data.loc[idx, 'Ethnicity'] = 1
    elif  row['Subject ID'] == 4:
        data.loc[idx, 'Ethnicity'] = 2
    else:
        data.loc[idx, 'Ethnicity'] = 0


# In[8]:

data['Age']= None
for idx, row in data.iterrows():
    if row['Subject ID'] == 2:
        data.loc[idx, 'Age'] = 31        
    elif  row['Subject ID'] == 3:
        data.loc[idx, 'Age'] = 28        
    else:
        data.loc[idx, 'Age'] = 24        


# In[9]:

data.head()


# In[10]:

plt.plot(data.Theta)


# In[13]:

sns.pairplot(data.iloc[:,2:13])


# In[ ]:

sns.pairplot(data[8:15])


# In[ ]:

# TO DO 

## Fill in some EDA
## Voting Ensemble
## Gradient Boosted Trees
## PCA
## Convert into Spyder

# Write-up

# Precision Recall Curve .. 


# In[14]:

train, test = train_test_split(data, test_size= .25, random_state=1)

train_x= train.drop(['userconfusion',"Subject ID","Video ID"], axis=1)
test_x= test.drop(['userconfusion',"Subject ID","Video ID"], axis=1)
train_y = train['userconfusion']
test_y = test['userconfusion']


# In[15]:

print("Baseline %")
sum(test_y)/len(test_y)*100


# ## DECISION TREE

# In[16]:

clf = DecisionTreeClassifier(max_depth=4)
clf.fit(train_x, train_y)
print("SCORE:")
clf.score(test_x, test_y) *100


# ## FEATURE IMPORTANCE

# In[17]:

clf.feature_importances_


# In[18]:

# Seems to potentially disprove the contention that 'Theta' waves might attend confusion
#col_headings = col_headings+['Gender','Ethnicity','Age']
feature_importance = pd.DataFrame(clf.feature_importances_, index=train_x.columns)
feature_importance.columns= ["Feature_Importance"]
feature_importance['Feature_Importance']
feature_importance= feature_importance.sort_values("Feature_Importance", axis = 0)


# In[19]:

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = feature_importance.index
y_pos = np.arange(len(feature_importance.index))
performance = feature_importance['Feature_Importance'] 
plt.barh(y_pos, performance, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('Importance %')
plt.title('Feature Importance Ranking')
plt.show()


# In[22]:

top_4_features = list(feature_importance['Feature_Importance'][11:].index)
top_4_features_response =  top_4_features+['userconfusion']
data_masked = data[top_4_features_response]
data_masked.head()


# In[23]:

# Masking dataframe to acount for top 4 features 
train, test = train_test_split(data_masked, test_size= .25, random_state=1)

train_x_masked= train.drop(['userconfusion'], axis=1)
test_x_masked= test.drop(['userconfusion'], axis=1)
train_y_masked = train['userconfusion']
test_y_masked = test['userconfusion']


# In[24]:

# LOSS LESS THAN 1% ACCURACY ...PROBABLY PREFERRABLE CONSIDERING SPARSE SOLUION
clf = DecisionTreeClassifier(max_depth=4)
clf.fit(train_x_masked, train_y_masked)
print("SCORE:")
clf.score(test_x_masked, test_y_masked) *100


# ## PIPELINE

# In[25]:

# With Standard Scaler
pipe_lr = Pipeline([('scl', StandardScaler()), ('clf', DecisionTreeClassifier())])


# In[26]:

print(pipe_lr.get_params())


# ## Validation Curves

# In[27]:

from sklearn.learning_curve import validation_curve 

param_range = [2, 4, 6, 8, 10, 12]
train_scores, test_scores = validation_curve(estimator=pipe_lr, X=train_x_masked, y=train_y_masked,
                                             param_name='clf__max_depth', param_range=param_range, cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)


# In[28]:

plt.plot(param_range, train_mean, label='Training_Accuracy')


# In[30]:

# Conclusion >> Max_Depth of 6 (or possibly 4) is ideal  .. Anything greater clear evidence of overfit
plt.plot(param_range, train_mean, label='Training_Accuracy')
plt.plot(param_range, test_mean, label='Testing_Accuracy')
plt.xlabel('Decision Tree Max Depth')
plt.legend()
plt.ylabel('Accuracy')


# ## GRID SEARCH

# In[31]:

gs = GridSearchCV(
    estimator= DecisionTreeClassifier(random_state=0),
    param_grid= [{'max_depth':[2,4,6,8,10,None]}],
    scoring= 'accuracy',
    cv= 10)

scores = cross_val_score(gs, train_x, train_y, scoring='accuracy', cv=10)


# In[32]:

print('CVAccuracy: %.3f+/- %.3f' %(np.mean(scores)*100, np.std(scores)))


# ## PERFORMANCE METRICS AND ROC 

# In[33]:

# Generate Confusion Matrix

print('Precision: %.3f', precision_score(y_true= test_y_masked, y_pred= clf.predict(test_x_masked))*100) 
print('Recall: %.3f', recall_score(y_true= test_y_masked, y_pred= clf.predict(test_x_masked)) *100)
print('F1: %.3f', f1_score(y_true= test_y_masked, y_pred= clf.predict(test_x_masked))*100) 
print('AUC: %.3f', roc_auc_score(y_true= test_y_masked, y_score= clf.predict(test_x_masked))*100)


# ## LOGISTIC REGRESSION

# In[34]:

lr_eeg=  LogisticRegression()
lr_eeg.fit(train_x, train_y)
lr_score= lr_eeg.score(train_x, train_y)*100
lr_score


# In[35]:

# Compute confusion matrix
print("Confusion Matrix ")
print("Classif: 0    1")
confusion_matrix(y_true=test_y, y_pred=lr_eeg.predict(test_x))


# ## LEARNING CURVE

# In[36]:

from sklearn.pipeline import Pipeline
from sklearn.learning_curve import learning_curve


train_sizes, train_scores, test_scores = learning_curve(LogisticRegression(penalty ='l2'),X= train_x, y= train_y, 
                                    train_sizes = [1000,2000,4000,6000,8000], cv= 10, n_jobs=1)
train_mean =  np.mean(train_scores, axis= 1)
train_std = np.std(train_scores, axis= 1)
test_mean = np.mean(test_scores, axis= 1)
test_std = np.std(test_scores, axis= 1)


# In[37]:

plt.plot(train_sizes, train_mean, label='Training_Accuracty')
plt.plot(train_sizes, test_mean, label='Testing_Accuracty')
plt.xlabel('Training Samples')
plt.legend()
plt.ylabel('Accuracy')


# In[512]:

clf.set_params(max_depth=2).fit(train_x_masked, train_y_masked)
dot_data= tree.export_graphviz(clf, out_file= None)

graph= pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("EEG_TREE_CLASSIFIER") 
Image(graph.create_png())


# ## RandomForest

# In[41]:

xlabels = []
n_trees = []
for i in range(50,500, 50):
    eeg_rf = RandomForestClassifier(n_estimators=i, max_features=4, random_state=1)
    eeg_rf.fit(train_x, train_y)
    n_trees.append(eeg_rf.score(test_x, test_y)*100)
    xlabels.append(i)


# In[536]:

# Implies 300 Trees Generates the Highest Accuracy
plt.plot(xlabels, n_trees)
plt.xlabel('Trees in RandomForest')
plt.ylabel('Accuracy')
plt.title("RandomForest Optimization")


# #RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, #min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, #oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)

# In[514]:

# Why does this decision path look like its got more features than 5 ?
print(eeg_rf.score(test_x, test_y)*100)
eeg_rf.decision_path(test_x)


# ## Ensemble Classifiers

# In[64]:

# Voting Classifier ... Accuracy lower than RandomForest (with 500 trees) ...possible ?
voting = VotingClassifier(estimators=[('lr', lr_eeg), ('rf', eeg_rf), ('gnb', clf)], voting='hard')
voting = voting.fit(train_x_masked, train_y_masked)
print("Confusion Matrix ")
print("Classif: 0    1")
print(confusion_matrix(y_true=test_y_masked, y_pred=voting.predict(test_x_masked)))
print()
print("Voting Classifier Accuracy: ")
print((1-(902+289)/(666+902+289+1346))*100)


# In[58]:

gbc= GradientBoostingClassifier(loss='deviance', learning_rate=0.1,
n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_split=1e-07, init=None, random_state=None,
max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')

print("Gradient Boosting Accuracy: ")
gbc.fit(train_x, train_y)
print(gbc.score(test_x, test_y)*100)


# In[114]:

scaler = StandardScaler()
X = scaler.fit_transform(train_x)
# variances given in descending order ...get labels
pca= PCA()
pca.fit_transform(X)                         
print(train_x.columns)
explained_variance= pca.explained_variance_ratio_
pca.explained_variance_ratio_


# In[128]:

with plt.style.context("dark_background"):
    plt.figure(figsize= (10,6))
    plt.bar(range(len(explained_variance)), explained_variance, alpha=.5, align= "center", label = "Individual Explained Variance")
#    plt.ylabel("Explained Variance")    
#    plt.xlabel("Principal Component")
    plt.legend(loc="best")
    plt.tight_layout()        


# In[78]:

pca_model = PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', 
                tol=0.0, iterated_power='auto', random_state=None)

train_x_scaled = scale(train_x)
train_y_scaled = scale(train_y)
#pca_model= pca_model.fit(pipe_lr)
pca_model= pca_model.fit(train_x_scaled, train_y_scaled)
print(train_x.columns)
pca_model.explained_variance_ratio_


# In[55]:

print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
lw = 2

# Binarize the output
y = label_binarize(test_y, classes=[0, 1])
# test_y.shape[1]
n_classes = 1

# Compute Precision-Recall and plot curve
precision = dict()
recall = dict()
average_precision = dict()


precision, recall, thresholds = precision_recall_curve(test_y, lr_score)


# Plot Precision-Recall curve
plt.clf()
plt.plot(recall, precision, lw=lw, color='navy',
         label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
#plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
plt.legend(loc="lower left")
plt.show()

