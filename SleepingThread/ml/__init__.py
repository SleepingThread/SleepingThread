# -*- coding: utf-8 -*-

"""

"""

import sys
import numpy as np

def weighted_accuracy(estimator,X,y):
    """
    y - two class prediction
    """
    y = np.array(y)
    prediction = estimator.predict(X)
    
    # work with labels
    labels = []
    for el in y:
        if el not in labels:
            labels.append(el)
            
    labels_qual = {el:0.0 for el in labels}
    labels_size = {el:np.sum(y==el) for el in labels}
    for i in xrange(len(y)):  
        labels_qual[y[i]] += (y[i]==prediction[i])
    
    qual = 0.0
    for key in labels_qual:
        qual += labels_qual[key]/labels_size[key]
    
    qual = qual/len(labels)
    
    return qual

def weighted_accuracy_score(y_true,y_pred):
    """
    """
    # work with labels
    labels = []
    for el in y_true:
        if el not in labels:
            labels.append(el)

    labels_qual = {el:0.0 for el in labels}
    labels_size = {el:np.sum(y_true==el) for el in labels}
    for i in xrange(len(y_true)):
        labels_qual[y_true[i]] += (y_pred[i]==y_true[i])

    qual = 0.0
    for key in labels_qual:
        qual += labels_qual[key]/labels_size[key]

    qual = qual/len(labels)

    return qual

def getNextVec(itvec,itend):
    """
    itvec,itend - array
    """
    for i in xrange(len(itvec)):
        itvec[i] += 1
        if itvec[i]<itend[i]:
            break
        else:
            itvec[i] = 0
   
    if not (i==len(itvec)-1 and itvec[-1]==0):
        return True
    else:
        return False

def getNextDict(itvec,itend,keys):
    """
    itvec,itend - dictionaries
    """
    i = 0
    for key in keys:
        i += 1
        itvec[key] += 1
        if itvec[key]<itend[key]:
            break
        else:
            itvec[key] = 0
   
    if not (i==len(keys) and itvec[keys[-1]]==0):
        return True
    else:
        return False

def getNext(itvec,itend,keys=None):
    if keys is None:
        return getNextVec(itvec,itend)
    else:
        return getNextDict(itvec,itend,keys)

    return

def ExecuteGrid(func,params):
    """
    params - dictionary
    """
    itvec = {key:0 for key in params}
    itend = {key:len(params[key]) for key in params}
    keys = params.keys()
    not_end = True
    while not_end:
        kwargs = {key:params[key][itvec[key]] for key in params}
        not_end = getNext(itvec,itend,keys)

        func(**kwargs)

    return

#========================================================
# Search best parameter with scree
#========================================================

from sklearn.base import clone

class ScreeSearch(object):
    def __init__(self,estimator,params,mode="quality"):
        """
        mode - "error" | "quality"
        params - {"param_name":list of values}
        """

        self.mode = mode

        # params list for estimator
        self.params = params
        self.estimator = estimator

        # store best trainer
        self.trainer_params = None
        self.trainer = None

        self.scores = []

        if len(params[params.keys()[0]]) <=3:
            raise Exception("Not enough params")

        return

    def fit(self,X,y=None,scoring=None):
        """
        scoring - dict with metrics(estimator,X,y)
        """

        self.scores = []
        param_name = self.params.keys()[0]
        for el in self.params[param_name]:
            cur_trainer = clone(self.estimator)
            args = {param_name:el}
            cur_trainer.set_params(**args)
            
            cur_trainer.fit(X,y)

            if scoring is None:
                self.scores.append(cur_trainer.score(X,y))
            else:
                scores = {}
                key = scoring.keys()[0]
                self.scores.append(scoring[key](cur_trainer,X,y))

        # find best parameter
        scree_vals = []
        for i in xrange(1,len(self.params[param_name])-1):
            a = self.scores[i-1]
            b = self.scores[i]
            c = self.scores[i+1]
            scree_vals.append(a+c-2*b)

        if self.mode == "error":
            # find max if self.scores - error
            best_ind = np.argmax(scree_vals)+1

        elif self.mode == "quality":
            # find min if self.scores - quality
            best_ind = np.argmin(scree_vals)+1

        self.trainer_params = {param_name:self.params[param_name][best_ind]}
        self.trainer = clone(self.estimator)
        self.trainer.set_params(**self.trainer_params)
        self.trainer.fit(X,y)

        return

    def predict(self,X):
        return self.trainer.predict(X)

    def drawScree(self):
        from matplotlib import pyplot as plt
        param_name = self.params.keys()[0]
        plt.plot(self.params[param_name],self.scores)
        plt.show()
        return

class GrowSpeed(object):
    def __init__(self,estimator,params,mode="quality",coef=0.5,verbose=0):
        """
        mode - "error" | "quality"
        params - {"param_name":list of values}
        """

        self.verbose = verbose

        self.coef = coef
        self.mode = mode

        # params list for estimator
        self.params = params
        self.estimator = estimator

        # store best trainer
        self.trainer_params = None
        self.trainer = None

        self.scores = []

        if len(params[params.keys()[0]]) <=3:
            raise Exception("Not enough params")

        return

    def fit(self,X,y=None,scoring=None):
        """
        scoring - dict with metrics(estimator,X,y)
        """

        self.scores = []
        param_name = self.params.keys()[0]
        params = self.params[param_name]

        _counter = 0
        _p_len = len(params)
        for el in params:
            _counter += 1
            if self.verbose > 0:
                sys.stdout.write("\rGrowSpeed: process "+str(_counter)+" from "+str(_p_len))

            cur_trainer = clone(self.estimator)
            args = {param_name:el}
            cur_trainer.set_params(**args)
            
            cur_trainer.fit(X,y)

            if scoring is None:
                self.scores.append(cur_trainer.score(X,y))
            else:
                scores = {}
                key = scoring.keys()[0]
                self.scores.append(scoring[key](cur_trainer,X,y))

        # find best parameter
        grow_vals = []
        for i in xrange(1,len(params)-1):
            a = self.scores[i-1]
            c = self.scores[i+1]
            grow_vals.append((c-a)/(params[i+1]-params[i-1]))

        if self.verbose > 0:
            sys.stdout.write("\n")

        if self.mode == "error" or self.mode == "quality":
            # find first which abs(.) less than 
            # abs((self.scores[-1]+self.scores[0])/(params[-1]-params[0])) 
            val = self.coef*abs((self.scores[-1]+self.scores[0])/(params[-1]-params[0]))
            for ind,el in enumerate(grow_vals):
                if abs(el)<val:
                    best_ind = ind + 1
                    break

        self.trainer_params = {param_name:self.params[param_name][best_ind]}
        self.trainer = clone(self.estimator)
        self.trainer.set_params(**self.trainer_params)
        self.trainer.fit(X,y)

        return

    def predict(self,X):
        return self.trainer.predict(X)

    def drawScores(self):
        from matplotlib import pyplot as plt
        param_name = self.params.keys()[0]
        plt.plot(self.params[param_name],self.scores)
        plt.show()
        return


#========================================================
# SVM with zero norm from Alexey Shestov <github: nelenivy>
#    Realization of Jason Weston "Use of the Zero-Norm with Linear Models and Kernel Methods"
#    Approximation of the zero-norm Minimization (AROM) - add multiplicative update
#    Use l2-SVM => l2-AROM
#    Regularization of linear inseparable case:
#       Make linear separation by hands - add new columns to kernel matrix
#
#    sci-kit learn trainer
#    
#    SVM_L0 - class for binary classification
#========================================================

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class SVM_L0(BaseEstimator,ClassifierMixin):
    def __init__(self,scale=True,C=1.0,eps=1.0e-3,\
            estimator=SVC(C=100000.0,kernel='linear'),verbose=0,\
            feature_selection=False,predict_estimator=SVC(C=1.0,kernel='linear'),\
            n_iter=1000):
        """
        feature_selection:
            True: use predict_estimator to predict data
            False: use 'z' coefs to predict data
        scale:
            don't supported, always acts like scale=True: uses StandardScaler
        """

        self.verbose = verbose

        # use algorithm for feature selection or for training
        self.feature_selection = feature_selection
        self.predict_estimator = predict_estimator
        self.trainer = None

        self.scale=scale
        self.scaler = StandardScaler()
        # parameter for regularization
        self.C = C

        # stop criterion parameters
        self.eps = eps
        self.n_iter = n_iter

        # estimator
        self.estimator = estimator

        # parameters for fit
        self.z = None
        self.inds = None
        self.intercept_ = None

        return
  
    def prepareGraphics(self):
        """
        """
        from matplotlib import pyplot as plt

        # initialize
        plt.ion()
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)

        return

    def drawZ(self,z_full,z_old_full,ymax=2.0):
        """
        """

        self.ax1.clear()
        self.ax2.clear()
        #self.ax1.set_ylim(0,ymax)
        self.ax1.bar(range(len(z_full)),z_full,color='blue')
        self.ax2.set_ylim(0,0.1)
        self.ax2.bar(range(len(z_full)),abs(z_full-z_old_full))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        return

    def fit(self,X,y):
        """
        Make SVML0 without matrix update (without removing unused features)
        """

        if self.verbose>=10:
            self.prepareGraphics()

        y = np.array(y)

        # y must have +1,-1 labels, check
        assert np.all((y==1)+(y==-1))

        # copy X, scale it , add regularization cols
        X = self.scaler.fit_transform(X)

        # regularization matrix
        _reg_mat = 1.0/self.C * np.ones((X.shape[0],X.shape[0]))

        # add regularization mat to data matrix
        X = np.concatenate((X,_reg_mat),axis=1)

        # selected columns
        #sel_inds = np.array(range(X.shape[1]))
        #sel_inds_prev = np.array(sel_inds)
        z = np.ones((1,X.shape[1]))
        z_prev = np.zeros((1,X.shape[1]))

        inds = np.array(range(X.shape[1]))

        trainer = None

        _counter = 0
        while not (np.all(abs(z-z_prev)<self.eps*np.max(z)) \
                or _counter>self.n_iter ):

            # get subset from previous subset
            subset = abs(z[0])>self.eps*np.max(z)
            z = z[:,subset]
            inds = inds[subset]

            # prepare matrix mat
            mat = X[:,inds]*z


            # calculate new z
            trainer = clone(self.estimator)
            # set trainer to be weighted
            trainer.set_params(class_weight={ 1:float(np.sum(y==-1))/float(np.sum(y==1)) })
            trainer.fit(mat,y)

            z_prev = z
            z = z*trainer.coef_


            if self.verbose>0:
                sys.stdout.write("\r"+str(_counter)+" | "+\
                        str(np.max(abs(z-z_prev)))+" | "+\
                        str(np.max(z)) + " | " + str(len(inds)) + " | ")
                sys.stdout.flush()

                if self.verbose >= 10:
                    # draw z full
                    _z_full = np.zeros(X.shape[1])
                    _z_full[inds] = z[0]
                    _z_old_full = np.zeros(X.shape[1])
                    _z_old_full[inds] = z_prev[0]
                    self.drawZ(_z_full,_z_old_full)

            _counter += 1 

        # remove regularizator inds
        subset = inds<(X.shape[1]-X.shape[0])
        inds = inds[subset]
        z = z[:,subset]

        self.inds = inds
        self.z = z
        self.intercept_ = trainer.intercept_
       
        if self.feature_selection:
            self.trainer = clone(self.predict_estimator)
            self.trainer.set_params(class_weight={ 1:float(np.sum(y==-1))/float(np.sum(y==1)) })
            mat = X[:,self.inds]
            self.trainer.fit(mat,y)

        #self.z = z[:,:(X.shape[1]-X.shape[0])]

        if self.verbose > 0:
            sys.stdout.write("\n")

        return

    def predict(self,X):
        """
        """
        X = self.scaler.transform(X)

        if not self.feature_selection:
            res = np.matmul(X[:,self.inds],self.z.T) + self.intercept_
            prediction = (res>0).astype(int)-(res<=0).astype(int)
        else:
            mat = X[:,self.inds]
            prediction = self.trainer.predict(mat)

        return prediction

#========================================================
# MGUA
#
#    sci-kit learn trainer
#========================================================

from sklearn.linear_model import LinearRegression

def addCol(arr,vec):
    """
    Change arr - add column
    """
    arr = np.concatenate((arr,vec.reshape(-1,1)),axis=1)
    return arr

class MGUABase(BaseEstimator):
    def __init__(self,Ct=0.8,n_comp=2,estimator=LinearRegression(),verbose=0,BUF_size=20,\
            init_method=1,predictor_type="regr"):
        """
        Ct - correlation threshold
        n_comp - number of components in linear model
            In oehter words - model complexity
        BUF_size - maximum number of elements in BUF
        init_method = 1 | 2
        predictor_type = "regr" | "binclass"
        """

        self.verbose = verbose

        self.init_method = init_method
        if init_method not in [1,2]:
            raise Exception("Unavailable init_method value")

        self.BUF1 = None
        self.BUF1coef_ = None
        self.BUF1intercept_ = None
        self.BUF1qual = None

        self.n_comp = n_comp
        self.Ct = Ct

        # for matrix scaling
        self.sc = StandardScaler()

        self.estimator = estimator

        self.coef_ = None
        self.intercept_ = None

        return

    def _isBetter(self,qual,BUFqual):
        """
        """
        result = []
        for el in BUFqual.T:
            _res = None
            for ind in xrange(len(qual)):
                if qual[ind]>el[ind]:
                    _res = True
                    break
                elif qual[ind]<el[ind]:
                    _res = False
                    break
            
            if _res is None:
                _res = False

            result.append(_res)

        return np.array(result) 

    def getQual(self,pred,target,coef_):
        """
        """
        from sklearn.metrics import r2_score

        if self.predictor_type == "regr":
            # calculate r2 score
            return np.array([r2_score(target,pred),1.0/(np.linalg.norm(coef_)+0.1)])
        elif self.predictor_type == "binclass":
            return np.array([weighted_quality_score(target,pred),\
                    1.0/(np.linalg.norm(coef_)+0.1)])
        else:
            raise Exception("Unknown predictor_type")

        return None

    def insertModel(self,BUF,BUFqual,BUFcoef_,BUFintercept_,pred,coef_,intercept_,target):
        """
        calculate variations 

        If no correlations - 
            select BUF_size best models

        If there are correlations - 
            add if <added model> has best quality
        """

        # calculate qual
        qual = self.getQual(pred,target,coef_)

        pred = pred-intercept_

        corr = abs(np.dot(BUF.T,pred)/(np.linalg.norm(BUF,axis=0)*np.linalg.norm(pred)))

        inds = corr>=self.Ct
        inds_add = inds+1

        res = self._isBetter(qual,BUFqual[inds])
        if np.all(res):
            # qual the best solution - remove BUFqual[inds]
            BUF = BUF[:,inds_add]
            BUFqual = BUFqual[:,inds_add]
            BUFcoef_ = BUFcoef_[:,inds_add]
            BUFintercept_ = BUFintercept_[inds_add]

            # check if there are space for new model
            if BUF.shape[1]<self.BUF_size:
                # add new model to buffers
                BUF = addCol(BUF,pred)
                BUFqual = addCol(BUFqual,qual)
                BUFcoef_ = addCol(BUFcoef_,coef_)
                BUFintercept_ = np.append(BUFintercept_,intercept_)
            else:
                # remove model with worst quality
                # find worst model:
                mininds = BUFqual[0,:]==np.min(BUFqual[0,:])
                minind2 = np.argmin(BUFqual[1,mininds])
                minind = mininds[minind2]

                BUF[:,minind] = pred
                BUFqual[:,minind] = qual
                BUFcoef_[:,minind] = coef_
                BUFintercept_[minind] = intercept_

        else:
            # ignore new model
            pass

        return

    def initBuffer(self,X,y,BUF1,BUF1params_):
        """
        """
        if self.init_method == 1:
            pass
        elif self.init_method == 2:
            for col1_ind in xrange(X.shape[1]-1):
                col1 = X[:,[col1_ind]]
                for col2_ind in xrange(col1_ind+1,X.shape[1]):
                    col2 = X[:,[col2_ind]]

                    mat = np.concatenate(col1,col2,axis=1)

                    trainer = clone(self.estimator)
                    trainer.fit(X,y)
                    pred = trainer.predict(X)

                    # trainer must have
                    self.insertModel(BUF1,BUF1coef_,BUF1intercept_,\
                            pred,trainer.coef_,trainer.intercept_)

        return

    def prefit(self,X,y):
        """
        Function created for set class_weights for estimators, which
            accept this property
        """
        #if hasattr(self.estimator,"class_weight"):
        #    self.estimator.set_params(class_weight={})
        return

    def fit(self,X,y,n_comp):
        """
        How to build first BUF1?:
            1. Build 1-complex model and continue
            2. Iterate over all available pairs of cols in X
        """

        self.prefit(X,y)

        # this step is not needed
        X = self.sc.fit_transform(X)

        # BUF1 - base buffer
        # BUF2 - buffer, that building now

        BUF1 = []
        BUF1params_ = []
        BUF2 = []
        BUF2params_ = []
 
        self.initBuffer(X,y,BUF1,BUF1coef_)

        if self.init_method == 1:
            n_steps = self.n_comp-1
        elif self.init_method == 2:
            n_steps = self.n_comp-2

        for it in xrange(n_steps):
            self.step(X,y,BUF1,BUF1coef_,BUF2,BUF2coef_)

        for it in xrange(self.n_comp):
            # build models with complexity it+1

            # copy BUF1 to BUF2
            BUF2 = BUF1
            BUF2coef_ = BUF1coef_

            # cycle: take col from X and col from BUF1
            for buf_el in BUF1:
                for col in X.T:
                    pass

            # add predicted to BUF2

        return

#========================================================
# Selectors
#========================================================
