import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression

#Group Method of Data Handling
class GMDH(BaseEstimator,ClassifierMixin):
    """
    Feature selection algorithm based on cross-validation
    """
    def __init__(self,trainer=LinearRegression,trainer_params=(),cross_val = ShuffleSplit ,cross_val_params=((),{})):
        BaseEstimator.__init__(self)
        ClassifierMixin.__init__(self)

        self.trainer_class = trainer
        self.trainer_params = trainer_params
        self.cross_val = cross_val
        self.cross_val_params = cross_val_params
        self.features_list = None
        self.trainer = None

        return

    def cv_score(self,data,target,trainer,trainer_params=(),cross_val = ShuffleSplit,cross_val_params = ((),{})):
        """
        return mean trainer score
        """
        score = 0.0
        trainer_class = trainer
        
        data = np.asarray(data)
        target = np.asarray(target)
        cv = cross_val(*cross_val_params[0],**cross_val_params[1])
       
        result = []
        for trainset, testset in cv.split(target):
            #prepare dataset
            data_train = data[trainset]
            target_train = target[trainset]
            data_test = data[testset]
            target_test = target[testset]

            #initialize trainer
            trainer = trainer_class(*trainer_params)
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
                trainer = self.trainer_class(*self.trainer_params)
                trainer.fit(newdata,target)
                cur_acc = trainer.score(newdata,target)

                if cur_acc > best_acc:
                    best_feature = fnum
                    best_acc = cur_acc

            #we found best feature, let's calculate cv score

            new_features = list(features_list)
            new_features.append(best_feature)

            #prepare new data - select features
            newdata = data[:,new_features]

            cv_score_new = self.cv_score(newdata,target,trainer = self.trainer_class,
                    trainer_params = self.trainer_params, cross_val = self.cross_val,cross_val_params = self.cross_val_params)

            cv_score_max = max(cv_score_max,cv_score_new[3])
            # print "cv_score_new: ",cv_score_new,len(new_features)
            if cv_score_new[3] <= cv_score_old[3]:
                #we cv value decrease
                #if cv_score_new[3]<0.97*cv_score_max:
                break
            
            features_list.append(best_feature)
            cv_score_old = cv_score_new

            if len(features_list) >= features_amount:
                #we select all features
                break

        #set self.features_list
        self.features_list = features_list

        #make appropriate trainer
        trainer = self.trainer_class(*self.trainer_params)
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

