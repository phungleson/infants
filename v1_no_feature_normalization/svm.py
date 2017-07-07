import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

from infants import deaths
from infants import births
from infants import X_COLUMNS

import logging
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

X1, y1 = deaths[X_COLUMNS], pd.Series([1] * 24174)
X2, y2 = births[X_COLUMNS], pd.Series([0] * 24175)

X1, y1 = X1[:1000], y1[:1000]
X2, y2 = X2[:1000], y2[:1000]

X, y = pd.concat([X1, X2]), pd.concat([y1, y2])

logging.info("X1=\n{}".format(X1[:10]))
logging.info("y1=\n{}".format(y1[:10]))
logging.info("X2=\n{}".format(X2[:10]))
logging.info("y2=\n{}".format(y2[:10]))
logging.info("X=\n{}".format(X[:10]))
logging.info("y=\n{}".format(y[:10]))

logging.info("Running SVM")

# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
clf = SVC(probability=True, random_state=0)
scores = cross_val_score(clf, X, y, cv=10, scoring='f1')

logging.info("Ran SVM")
logging.info("Mean F1 {:0.2f} (+/- {:0.2f})".format(scores.mean(), scores.std() * 2))
