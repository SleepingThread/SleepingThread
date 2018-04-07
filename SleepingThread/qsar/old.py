
class Net0(gluon.Block):
    def __init__(self,**kwargs):
        super(Net0, self).__init__(**kwargs)
        with self.name_scope():
            self.dense = gluon.nn.Dense(1)
        return

    def forward(self, x):
        x = self.dense(x)
        return x



"""
class Net2(gluon.Block):
    def __init__(self,**kwargs):
        super(Net2, self).__init__(**kwargs)
        with self.name_scope():
            self.conv0 = gluon.nn.Conv2D(1,3)
            self.conv1 = gluon.nn.Conv2D(1,3)
            self.conv2 = gluon.nn.Conv2D(1,3)
            self.conv3 = gluon.nn.Conv2D(1,3)

            self.drop0 = gluon.nn.Dropout(0.4)
            self.drop1 = gluon.nn.Dropout(0.4)
            self.drop2 = gluon.nn.Dropout(0.4)
            self.drop3 = gluon.nn.Dropout(0.4)

            self.pool0 = gluon.nn.MaxPool2D(2)
            self.pool1 = gluon.nn.MaxPool2D(2)
            self.pool2 = gluon.nn.MaxPool2D(2)
            self.pool3 = gluon.nn.MaxPool2D(2)

            self.flat = gluon.nn.Flatten()
            self.dense = gluon.nn.Dense(1)

            #self.conv = [self.conv0,self.conv1,self.conv2,self.conv3]
            #self.drop = [self.drop0,self.drop1,self.drop2,self.drop3]
            #self.pool = [self.pool0,self.pool1,self.pool2,self.pool3]

            #self.verb = 1

        return

    def forward(self, x):
        #print "FORWARD: "
       
        x = nd.relu(self.conv0(x))
        x = self.drop0(x)
        x = self.pool0(x)

        x = nd.relu(self.conv1(x))
        x = self.drop1(x)
        x = self.pool1(x)

        x = nd.relu(self.conv2(x))
        x = self.drop2(x)
        x = self.pool2(x)

        x = nd.relu(self.conv2(x))
        x = self.drop2(x)
        x = self.pool2(x)

        print x.shape
        
        x = self.flat(x)
        x = self.dense(x)

        return x
"""

class Net5(gluon.Block):
    def __init__(self,**kwargs):
        super(Net5,self).__init__(**kwargs)
        with self.name_scope():
            self.dense = gluon.nn.Dense(1)

        return 

    def forward(self,x):
        x = nd.relu(self.dense(x))

        return x


def testNet5():
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
    
    net = Net5()
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

def testNet(data,target,num_inputs):
    batch_size = 20
    num_outputs = 1
    num_examples = len(target)

    model_ctx = mx.cpu()

    data = nd.array(data).reshape((-1,num_inputs*num_inputs))
    target = nd.array(target)

    print "Data shape: ",data.shape
    print "Target shape: ",target.shape

    net = Net5()
    net.collect_params().initialize(mx.init.Normal(sigma=0.1))

    l2loss = gluon.loss.L2Loss()
    trainer = gluon.Trainer(net.collect_params(),'adam')

    random_sampler = gluon.data.RandomSampler(num_examples)
    sampler = gluon.data.BatchSampler(random_sampler,batch_size,'keep')

    epochs = 10
    
    for e in xrange(epochs):
        print "Epoch ",e
        cum_loss = 0.0
        for i,idx in enumerate(sampler):
            cur_data = data[idx]
            cur_labels = target[idx]
            print "data: ",cur_data.shape
            print "labels: ",cur_labels.shape
            with autograd.record():
                output = net(cur_data)
                loss_val = l2loss(output,cur_labels)
            loss_val.backward()
            trainer.step(cur_data.shape[0])
            cum_loss += nd.sum(loss_val).asscalar()
        #if e%10 == 0:
        #    print "Epoch %s loss: %s"%(str(e),str(cum_loss))
            
    #print "Epoch %s loss: %s"%(str(epochs-1),str(cum_loss))

    #output = net(data)
    #print "Result Mean Loss: ",nd.sum(l2loss(output,target)).asscalar()/num_examples
    
    return
