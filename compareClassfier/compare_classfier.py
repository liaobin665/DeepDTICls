# 功能：收集不同 算法的，不同性能指标，如：accuracy	precision	recall	F1-score
#导入机器学习算法库
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from xgboost import XGBClassifier
from sklearn.model_selection import  train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_curve, precision_score, accuracy_score,recall_score, f1_score,roc_auc_score,classification_report,cohen_kappa_score
import numpy as np
# 读取带标签的数据

# X= pd.read_csv('test_middle_X.csv').loc[:,'0':'255']
# y = pd.read_csv('../dataPubChem/pubchem_y_test.csv')

X= pd.read_csv('train_middle_X.csv').loc[:,'0':'255']
y = pd.read_csv('../dataPubChem/pubchem_y_train.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0)

#汇总不同模型算法
classifiers=[]

classifiers.append(DecisionTreeClassifier())
classifiers.append(RandomForestClassifier())
classifiers.append(ExtraTreesClassifier())
classifiers.append(GradientBoostingClassifier())
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression())
classifiers.append(LinearDiscriminantAnalysis())
classifiers.append(MLPClassifier())
classifiers.append(SVC())
classifiers.append(XGBClassifier())

accuracy=[]
precision=[]
recall=[]
F1_score=[]
roc_auc=[]
kappa = []
AUCs = []
AUPRs = []

for model in classifiers:
    model.fit(X_train,y_train)
    y_score = model.predict(X_test)
    fpr, tpr, thresholds_AUC = roc_curve(y_test, y_score)

    AUCs.append(auc(fpr,tpr))
    accuracy.append(accuracy_score(y_test,y_score))
    precision.append(precision_score(y_test,y_score,average="macro"))
    recall.append((recall_score(y_test,y_score,average="macro")))
    pre, re, thr =precision_recall_curve(y_test,y_score)
    AUPR = auc(re,pre)
    AUPRs.append(AUPR)

    F1_score.append(f1_score(y_test,y_score,average="macro"))
    roc_auc.append(roc_auc_score(y_test,y_score))
    kappa.append(cohen_kappa_score(y_test,y_score))
    ans = classification_report(y_test, y_score, digits=5)
    print(str(model)+"  :-------------classification_report------------->>>>>")
    print(ans)

print("-----------------------------------------------------------------------------")
# 汇总数据
cvResDf = pd.DataFrame({'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score':F1_score,
                        'roc_auc':roc_auc,
                        'kappa_score':kappa,
                        'AUCs':AUCs,
                        'AUPRs':AUPRs,
                        'algorithm': ['DecisionTreeClassifier','RandomForestClassifier','ExtraTreesClassifier','GradientBoostingClassifier',
                                      'KNeighborsClassifier','LogisticRegression','LinearDiscriminantAnalysis','MLPClassifier','SVC','XGBClassifier']})
print(cvResDf)
cvResDf.to_csv("train_compare_classification_report222.csv")

