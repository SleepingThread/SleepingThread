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

#========================================================
# Selectors
#========================================================
