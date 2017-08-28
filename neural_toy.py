import numpy as np

class simple_neural_network_clf(object):
    def __init__(self,n_node,n_epoch,activation,n_input_dim,n_output_dim,learning_rate,random_state):
        self.num_node = n_node
        self.num_input_dim = n_input_dim
        self.num_output_dim = n_output_dim
        self.num_epoch = n_epoch
        self.activation = activation
        if self.activation!='sigmoid':
            raise ValueError('>>currently only support sigmoid activation function')
        self.rate = learning_rate
        #random init all parameters
        np.random.seed(random_state)
        self.v = np.random.randn(self.num_input_dim, self.num_node)
        self.b = np.zeros((1, self.num_node))
        self.w = np.random.randn(self.num_node, self.num_output_dim)
        self.a = np.zeros((1, self.num_output_dim))

    def sigmoid(self, feed):
        return 1.0 / (1 + np.exp(-1.0*feed))

    def forward_propagation(self,x,v,b,w,a):
        z1 = x.dot(v) + b
        h = self.sigmoid(z1)
        z2 = h.dot(w) + a
        output = self.sigmoid(z2)
        return {'output':output,'h':h}


    def back_propagation(self,x,y,y_hat,v,b,w,a,h):
        g = y_hat*(1-y_hat)*(y_hat-y)
        dw = (g.T.dot(h)).T
        da = np.sum(y_hat*(1-y_hat)*(y_hat-y),axis=0)
        dv = ((h*(1-h)).T*w.dot(g.T)).dot(x).T
        db = np.sum((h*(1-h)).T*w.dot(g.T),axis=1)
        return {'dv':dv,'db':db,'dw':dw,'da':da}



    def build_model(self,x,y,print_loss=True,print_step=1000):
        if x.shape[1] != self.num_input_dim:
            raise ValueError('>>input data dimention not equal to n_input_dim')
        if y.shape[1] != self.num_output_dim:
            raise ValueError('>>output data dimention not equal to n_output_dim')
        self.model = {}
        size = float(x.shape[0])
        for i in range(self.num_epoch):
            fp = self.forward_propagation(x,self.v,self.b,self.w,self.a)
            y_hat = fp['output']
            h = fp['h']
            delta = self.back_propagation(x,y,y_hat,self.v,self.b,self.w,self.a,h)
            self.v -= self.rate * delta['dv']/size
            self.b -= self.rate * delta['db']/size
            self.w -= self.rate * delta['dw']/size
            self.a -= self.rate * delta['da']/size

            if print_loss and i%print_step == 0:
                point_error = np.sum((y_hat-y)**2,axis=1)/2.0
                accumulated_error = np.sum(point_error,axis=0)/size
                pred = np.empty((0,self.num_output_dim))
                for j in y_hat:
                    pred_init = np.zeros((1, self.num_output_dim))
                    pred_init[0][np.argmax(j)] = 1
                    pred = np.append(pred,pred_init,axis=0)
                accuracy = 1 - np.count_nonzero(np.sum(np.abs(pred - y),axis=1))/size
                print(">>Loss after iteration %i: %f \t Accuracy: %f"%(i, accumulated_error,accuracy))

    def predict(self, feed):
        y_hat = self.forward_propagation(feed,self.v,self.b,self.w,self.a)['output']
        pred = np.empty((0,self.num_output_dim))
        for j in y_hat:
            pred_init = np.zeros((1, self.num_output_dim))
            pred_init[0][np.argmax(j)] = 1
            pred = np.append(pred,pred_init,axis=0)
        return pred
