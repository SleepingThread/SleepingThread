import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.linear_model import LinearRegression

#Group Method of Data Handling
class GMDH(BaseEstimator,RegressorMixin):
    """
    Feature selection algorithm based on cross-validation
    """
    def __init__(self,estimator=LinearRegression(),\
            cv=ShuffleSplit(n_splits=20,test_size=0.2,random_state=0),max_features=6):

        self.estimator = estimator
        self.cv = cv
        self.max_features = max_features

        self.features_list = None
        self.trainer = None

        return

    def cv_score(self,data,target):
        """
        return mean trainer score
        """
        score = 0.0
        
        data = np.asarray(data)
        target = np.asarray(target)
        cv = self.cv
       
        result = []
        for trainset, testset in cv.split(target):
            #prepare dataset
            data_train = data[trainset]
            target_train = target[trainset]
            data_test = data[testset]
            target_test = target[testset]

            #initialize trainer
            trainer = clone(self.estimator)
            trainer.fit(data_train,target_train)
            result.append(trainer.score(data_test,target_test))
            #score += trainer.score(data_test,target_test)

        #score = score / cv.get_n_splits(target)

        return (np.average(result),np.min(result),np.max(result),np.median(result))

    def qual(self,cv_score):
        return cv_score[0]+0.06*cv_score[1]-0.02*cv_score[2]

    def fit(self,data,target):
        """
        Find best compexity with crossvalidation
        Find best model for best complexity
        """

        features_list = []

        #prepare dataset, use pandas
        data = np.asarray(data)
        target = np.asarray(target)

        features_amount = data.shape[1]

        cv_score_max = float('-inf')
        cv_score_new = 0.0
        cv_score_old = [float('-inf')] * 4

        while True:

            #select feature with best model accuracy
            best_feature = None
            best_acc = float('-inf')
            for fnum in xrange(features_amount):
                if fnum in features_list:
                    continue

                new_features = list(features_list)
                new_features.append(fnum)

                newdata = data[:,new_features]

                #train model
                #trainer = clone(self.estimator)
                #trainer.fit(newdata,target)
                #cur_acc = trainer.score(newdata,target)
                cur_acc = np.average(cross_val_score(self.estimator,newdata,target))

                if cur_acc > best_acc:
                    best_feature = fnum
                    best_acc = cur_acc

            """
            #we found best feature, let's calculate cv score

            new_features = list(features_list)
            new_features.append(best_feature)

            #prepare new data - select features
            newdata = data[:,new_features]

            cv_score_new = self.cv_score(newdata,target)

            cv_score_max = max(cv_score_max,cv_score_new[3])
            # print "cv_score_new: ",cv_score_new,len(new_features)
            if cv_score_new[3] <= cv_score_old[3]:
                #we cv value decrease
                #if cv_score_new[3]<0.97*cv_score_max:
                break
            """
            features_list.append(best_feature)
            cv_score_old = cv_score_new

            if len(features_list) >= features_amount or \
                    len(features_list) >= self.max_features:
                #we select all features
                break

        self.best_acc = best_acc

        #print "best_acc: ",best_acc
        #print "features_list: ",features_list

        #set self.features_list
        self.features_list = features_list

        #make appropriate trainer
        trainer = clone(self.estimator)
        new_data = data[:,features_list]
        trainer.fit(new_data,target)
        self.trainer = trainer

        return

    def predict(self,data):
        if self.features_list is None or self.trainer is None:
            raise ValueError("Need to use fit before predict")

        #prepare data
        data = np.asarray(data)

        new_data = data[:,self.features_list]

        return self.trainer.predict(new_data)

    def score(self,data,target):
        if self.features_list is None or self.trainer is None:
            raise ValueError("Need to use fit before score")

        #prepare data
        data = np.asarray(data)

        new_data = data[:,self.features_list]

        return self.trainer.score(new_data,target)

