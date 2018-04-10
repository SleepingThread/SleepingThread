from SleepingThread.st_gmdh import GMDH
from sklearn.model_selection import ShuffleSplit
from sklearn.base import clone

gmdh = GMDH(cv=ShuffleSplit(n_splits=200,test_size=0.2,random_state=0))

