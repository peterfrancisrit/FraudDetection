import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Import the data
raw_data = pd.read_csv('../creditcard.csv')


# Log transform on the data, keep both datasets. Use log for logistic and the features 
# for the booster or SVM

data_norm = raw_data[raw_data['Class'] == 0]
data_norm = raw_data.drop(['Class'],axis=1)

clf = OneClassSVM()

param_dist = {
              'kernel':['rbf']}

# run randomized search
n_iter_search = 100
random_search = GridSearchCV(clf, param_grid=param_dist,
                                    cv=3, scoring='accuracy',iid=False)
random_search.fit(data_norm)
res = pd.DataFrame(random_search.cv_results_)
print(res)


