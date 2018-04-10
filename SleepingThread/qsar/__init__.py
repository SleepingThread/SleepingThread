# -*- coding: utf-8 -*-
"""
module qsar
"""

import os
import copy
import random
import numpy as np

# sklearn import 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut, ShuffleSplit
from sklearn.base import BaseEstimator, RegressorMixin

# mxnet block
import mxnet as mx
from mxnet import gluon, nd, autograd

iteration = 0
pred_result = []

class FilenameIter(object):
    """!
    @brief Brief description
    """
    def __init__(self, path, ext, start_index=1, n_files=-1):
        if isinstance(ext,str):
            self.ret_type = "str"
            ext = [ext]
        else:
            self.ret_type = "list"

        self.path = path
        self.ext = ext
        self.start_index = start_index
        self.n_files = n_files
        return

    def __iter__(self):
        cur_index = self.start_index-1
        while True:
            cur_index += 1
            res = []
            for ext in self.ext:
                filename = self.path+"/"+str(cur_index)+"."+ext
                if os.path.isfile(filename):
                    res.append(filename)                    

            if len(res) == 0:
                break

            if len(res) != len(self.ext):
                raise Exception("len("+str(res)+") != len("+str(self.ext)+")")

            if self.ret_type != "str":
                yield res
            else:
                yield res[0]

        return

def test_FilenameIter():
    path = "/home/unknown/SCW/datasets/1/selection"
    fileiter = FilenameIter(path, ["mol","points","meshidx"])
    for res in fileiter:
        print res
    return


#========================================================
# Neural network and trainers part
#========================================================


class Net1(gluon.Block):
    def __init__(self,**kwargs):
        super(Net1, self).__init__(**kwargs)
        with self.name_scope():
            self.flat = gluon.nn.Flatten()
            self.dense = gluon.nn.Dense(1)
            """
            self.conv0 = gluon.nn.Conv2D(1,5)
            self.pool0 = gluon.nn.MaxPool2D(4)
            self.drop0 = gluon.nn.Dropout(0.4)
            self.conv1 = gluon.nn.Conv2D(1,5)
            self.pool1 = gluon.nn.MaxPool2D(4)
            self.drop1 = gluon.nn.Dropout(0.4)
            self.flat2 = gluon.nn.Flatten()
            self.dense2 = gluon.nn.Dense(1)
            #self.dense1 = gluon.nn.Dense(64)
            #self.dense2 = gluon.nn.Dense(10)
            #self.drop = gluon.nn.Dropout(0.5)
            """
            #self.verb = 1
        return

    def forward(self, x):
        x = self.flat(x)
        x = self.dense(x)
        #print "FORWARD: "
        #x = nd.relu(self.conv0(x))
        #x = self.drop0(x)
        #print x.shape
        #x = self.pool0(x)
        #print x.shape
        #x = self.drop0(x)
        #print x.shape
        #x = nd.relu(self.conv1(x))
        #x = self.drop1(x)
        #print x.shape
        #x = self.pool1(x)
        #print x.shape
        #x = self.flat2(x)
        #print x.shape
        #x = self.dense2(x)
        #print x.shape
        return x


class Net2(gluon.Block):
    def __init__(self,**kwargs):
        super(Net2, self).__init__(**kwargs)
        with self.name_scope():
            self.conv0 = gluon.nn.Conv2D(1,3)
            self.pool0 = gluon.nn.MaxPool2D(2)
            #self.drop0 = gluon.nn.Dropout(0.4)

            self.conv1 = gluon.nn.Conv2D(1,3)
            #self.drop1 = gluon.nn.Dropout(0.4)
            
            self.flat2 = gluon.nn.Flatten()
            self.dense2 = gluon.nn.Dense(1)
            
            #self.dense1 = gluon.nn.Dense(64)
            #self.dense2 = gluon.nn.Dense(10)
            #self.drop = gluon.nn.Dropout(0.5)
            #self.verb = 1
        return

    def forward(self, x):
        x = self.conv0(x)
        x = self.pool0(x)
        #x = self.drop0(x)

        x = self.conv1(x)
        #x = self.drop1(x)

        x = self.flat2(x)
        x = self.dense2(x)
        
        return x


#class Trainer(BaseEstimator,ClassifierMixin)
class Trainer(BaseEstimator,RegressorMixin):
    def __init__(self,num_inputs,mxseed=0,epochs=5000,net_type=1):
        BaseEstimator.__init__(self)
        RegressorMixin.__init__(self)
        self.net = None
        self.num_inputs = num_inputs
        self.mxseed = mxseed
        self.epochs = epochs
        self.net_type = net_type
        return

    def fit(self,data,target):
        
        global pred_result, iteration

        iteration += 1
        print "FIT CALL №",iteration
        
        batch_size = 20
        num_inputs=self.num_inputs
        num_outputs = 1
        num_examples = len(target)

        data = nd.array(data).reshape((-1,1,num_inputs,num_inputs))
        target = nd.array(target)

        mx.random.seed(self.mxseed)
        random.seed(self.mxseed)

        if self.net_type == 1:
            self.net = Net1()
        elif self.net_type == 2:
            self.net = Net2()
        else:
            raise Exception("No such net_type")

        net = self.net
        net.collect_params().initialize(mx.init.Normal(sigma=0.1))

        l2loss = gluon.loss.L2Loss()
        trainer = gluon.Trainer(net.collect_params(),'adam')

        random_sampler = gluon.data.RandomSampler(num_examples)
        #random_sampler = gluon.data.SequentialSampler(num_examples)
        sampler = gluon.data.BatchSampler(random_sampler,batch_size,'keep')

        epochs = self.epochs

        for e in xrange(epochs):
            cum_loss = 0.0
            for i,idx in enumerate(sampler):
                cur_data = data[idx]
                cur_labels = target[idx]
                with autograd.record():
                    output = net(cur_data)
                    loss_val = l2loss(output,cur_labels)
                loss_val.backward()
                trainer.step(cur_data.shape[0])
                cum_loss += nd.sum(loss_val).asscalar()
            if e%100 == 0:
                print "Epoch %s loss: %s"%(str(e),str(cum_loss))
                
        
        #print "Epoch %s loss: %s"%(str(epochs-1),str(cum_loss))
    
        output = net(data)
        print "Result Mean Loss: ",nd.sum(l2loss(output,target)).asscalar()/num_examples       
        pred_result.append([target.asnumpy(),output.asnumpy()])

        return

    def predict(self,data):
        data = nd.array(data).reshape((-1,1,self.num_inputs,self.num_inputs))
        output = self.net(data)
        return output.asnumpy()

    def score(self,X,y,sample_weight=None):
        if len(X) == 1:
            output = self.predict(X)
            return output.shape[0]*mean_squared_error(y,output)
        
        return RegressorMixin.score(self,X,y,sample_weight=sample_weight)

def testTrainer(data,target):
    trainer = Trainer()
    trainer.fit(data,target)
    return trainer.score(data,target)



"""
class Net1(gluon.Block):
    def __init__(self,**kwargs):
        super(Net1,self).__init__(**kwargs)
        with self.name_scope():
            self.dense = gluon.nn.Dense(1)

        return 

    def forward(self,x):
        x = nd.relu(self.dense(x))

        return x

def testNet():
    import pickle
    #import data 
    fin = open("data52","rb")
    data = pickle.load(fin)
    fin.close()
    
    target = nd.array(range(104))
    
    sampler = gluon.data.SequentialSampler(104)
    data = np.asarray(data)
    data = data[:,1]
    data = nd.array(data).reshape((-1,52*52))
    
    net = Net1()
    net.collect_params().initialize(mx.init.Normal(sigma=0.1))
    l2loss = gluon.loss.L2Loss()
    trainer = gluon.Trainer(net.collect_params(),'adam')
    
    output = net(data)
    
    for e in xrange(10):
        cum_loss = 0.0
        print "Epoch"
        for i,idx in enumerate(sampler):
            cur_data = data[[idx]]
            cur_target = target[[idx]]
    
            with autograd.record():
                output = net(cur_data)
                loss_val = l2loss(output,cur_target)
    
            loss_val.backward()
            trainer.step(cur_data.shape[0])
            cum_loss += nd.sum(loss_val).asscalar()

    return
"""      
