import numpy as np

import mxnet as mx
from mxnet import gluon, nd, autograd

class Net1(gluon.Block):
    def __init__(self,**kwargs):
        super(Net1,self).__init__(**kwargs)
        with self.name_scope():
            self.dense = gluon.nn.Dense(1)

        return 

    def forward(self,x):
        x = nd.relu(self.dense(x))

        return x


net = Net1()

data = np.array(range(100))
target = 1.24*np.array(range(100))+10.0

data = nd.array(data)
target = nd.array(target)

epochs=2

sampler = gluon.data.SequentialSampler(100)

net.collect_params().initialize(mx.init.Normal(sigma=0.1))

l2loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(),'adam')

for e in xrange(epochs):
    cum_loss = 0.0
    for i,idx in enumerate(sampler):
        cur_data = data[idx]
        cur_target = target[idx]

        with autograd.record():
            output = net(cur_data)
            loss_val = l2loss(output,cur_target)

        loss_val.backward()
        trainer.step(cur_data.shape[0])
        cum_loss += nd.sum(loss_val).asscalar()

    print "Epoch ",e," loss: ",cum_loss



