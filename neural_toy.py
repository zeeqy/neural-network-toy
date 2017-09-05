import numpy as np

class nn_clf(object):
    #TODO: add more activation functions
    #add batch and minibatch training schema
    #support multilayer traning
    
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
        
        ######parameters#####
        #hidden layer weights
        self.v = np.random.randn(self.num_input_dim, self.num_node)
        #hidden layer bias
        self.hbias = np.zeros((1, self.num_node))
        #output layer weights
        self.w = np.random.randn(self.num_node, self.num_output_dim)
        #output layer bias
        self.obias = np.zeros((1, self.num_output_dim))

    def sigmoid(self,t):
        return 1.0 / (1 + np.exp(-1.0*t))

    def forward_propagation(self,x,v,hbias,w,obias):
        z1 = x.dot(v) + hbias
        h = self.sigmoid(z1)
        z2 = h.dot(w) + obias
        output = self.sigmoid(z2)
        return {'output':output,'h':h}


    def back_propagation(self,x,y,y_hat,v,hbias,w,obias,h):
        g = y_hat*(1-y_hat)*(y_hat-y)
        d_w = (g.T.dot(h)).T
        d_obias = np.sum(g,axis=0)
        d_v = ((h*(1-h)).T*w.dot(g.T)).dot(x).T
        d_hbias = np.sum((h*(1-h)).T*w.dot(g.T),axis=1)
        return {'d_v':d_v,'d_hbias':d_hbias,'d_w':d_w,'d_obias':d_obias}



    def fit(self,x,y,print_loss=True,print_step=1000):
        if x.shape[1] != self.num_input_dim:
            raise ValueError('>>input data dimention not equal to n_input_dim')
        if y.shape[1] != self.num_output_dim:
            raise ValueError('>>output data dimention not equal to n_output_dim')
            
        self.size = float(x.shape[0])
        
        for i in range(self.num_epoch):
            self.model = {'v':self.v,'hbias':self.hbias,'w':self.w,'obias':self.obias}
            forward_propagation = self.forward_propagation(x,self.v,self.hbias,self.w,self.obias)
            y_hat = forward_propagation['output']
            h = forward_propagation['h']
            
            delta = self.back_propagation(x,y,y_hat,self.v,self.hbias,self.w,self.obias,h)
            self.v -= self.rate * delta['d_v']/self.size
            self.hbias -= self.rate * delta['d_hbias']/self.size
            self.w -= self.rate * delta['d_w']/self.size
            self.obias -= self.rate * delta['d_obias']/self.size

            if print_loss and i%print_step == 0:
                point_error = np.sum((y_hat-y)**2,axis=1)/2.0
                accumulated_error = np.sum(point_error,axis=0)/self.size
                pred = np.empty((0,self.num_output_dim))
                for j in y_hat:
                    pred_init = np.zeros((1, self.num_output_dim))
                    pred_init[0][np.argmax(j)] = 1
                    pred = np.append(pred,pred_init,axis=0)
                accuracy = 1 - np.count_nonzero(np.sum(np.abs(pred - y),axis=1))/self.size
                print(">>Loss after iteration %i: %f \t Accuracy: %f"%(i, accumulated_error,accuracy))

    def predict(self, x):
        y_hat = self.forward_propagation(x,self.v,self.hbias,self.w,self.obias)['output']
        pred = np.empty((0,self.num_output_dim))
        for j in y_hat:
            pred_init = np.zeros((1, self.num_output_dim))
            pred_init[0][np.argmax(j)] = 1
            pred = np.append(pred,pred_init,axis=0)
        return pred
    
    
    
class binary_rbm(object):
    def __init__(self, n_node, n_epoch, n_input_dim, learning_rate, random_state):
        
        self.num_node = n_node
        self.rate = learning_rate
        self.num_input_dim = n_input_dim
        self.num_epoch = n_epoch
        
        np.random.seed(random_state)

        #hidden layer weights
        self.v = np.random.randn(self.num_input_dim, self.num_node)
        #hidden layer bias
        self.hbias = np.zeros((1, self.num_node))
        #visible layer bias
        self.vbias = np.zeros((1, self.num_input_dim))

    def sigmoid(self, t):
        return 1.0 / (1 + np.exp(-1.0*t))
    
    def propup(self, x):
        z1 = x.dot(self.v) + self.hbias
        return self.sigmoid(z1)

    def propdown(self, h):
        z2 = h.dot(self.v.T) + self.vbias
        return self.sigmoid(z2)
    
    def gibbs_hvh(self,h_state):
        "hidden to visible"
        v = self.propdown(h_state)
        v_state = np.random.binomial(size=v.shape,n=1,p=v)
        
        "then visible to hidden"
        h = self.propup(v_state)
        h_state = np.random.binomial(size=h.shape,n=1,p=h)
        
        return v_state,h_state
        
    
    def fit(self,x,k=1,print_loss=True,print_step=1000):
        if x.shape[1] != self.num_input_dim:
            raise ValueError('>>input data dimention not equal to n_input_dim')
        
        self.size = float(x.shape[0])
        
        for i in range(self.num_epoch):
            self.model = {'v':self.v,'hbias':self.hbias,'vbias':self.vbias}
            
            h = self.propup(x)
            h_state = np.random.binomial(size=h.shape,n=1,p=h)
            
            "CD-k, machine dreaming..."
            for step in range(k):
                if step == 0:
                    nx_samples,nh_samples = self.gibbs_hvh(h_state)
                else:
                    nx_means, nh_samples = self.gibbs_hvh(nh_samples)
            
            self.v += self.rate * (x.T.dot(h) - nx_samples.T.dot(nh_samples))/self.size
            self.hbias += self.rate * np.mean(h - nh_samples,axis=0)
            self.vbias += self.rate * np.mean(x - nx_samples,axis=0)
            
            if print_loss and i%print_step == 0:
                monitoring_cost = np.mean((x - nx_samples)**2)
                print(">>Loss after iteration %i: %f"%(i, monitoring_cost))
                
    def reconstruct(self, x):
        h = self.propup(x)
        reconstructed_x = self.propdown(h)
        return reconstructed_x
    
    def transfer(self,x):
        h = self.propup(x)
        return np.random.binomial(size=h.shape,n=1,p=h)

                
class guassian_rbm(object):
    def __init__(self, n_node, n_epoch, n_input_dim, sigma, weight_learning_rate,slop_learning_rate, random_state):
        
        self.num_node = n_node
        self.weight_rate = weight_learning_rate
        self.slop_rate = slop_learning_rate
        self.num_input_dim = n_input_dim
        self.num_epoch = n_epoch
        
        np.random.seed(random_state)
        #hidden layer weights
        self.v = np.random.randn(self.num_input_dim, self.num_node)
        #hidden layer bias
        self.hbias = np.ones((1, self.num_node))
        #visible layer bias
        self.vbias = np.ones((1, self.num_input_dim))
        #steepness of the sigmoid function
        self.hslop = np.ones((1, self.num_node))
        self.vslop = np.ones((1, self.num_input_dim))
        
        self.sig = sigma

    def sigmoid(self, t):
        return 1.0 / (1 + np.exp(-1.0*t))
    
    def propup(self, x):
        z1 = self.hslop*(x.dot(self.v) + self.hbias + self.sig*np.random.normal(size=self.num_node))
        #return np.min(x,axis=1).reshape(x.shape[0],1) + (np.max(x,axis=1) - np.min(x,axis=1)).reshape(x.shape[0],1) * self.sigmoid(z1)
        return self.sigmoid(z1) 

    def propdown(self, h):
        z2 = self.vslop*(h.dot(self.v.T) + self.vbias + self.sig*np.random.normal(size=self.num_input_dim))
        #return np.min(h,axis=1).reshape(h.shape[0],1) + (np.max(h,axis=1) - np.min(h,axis=1)).reshape(h.shape[0],1) * self.sigmoid(z2)
        return self.sigmoid(z2) 
    
    def gibbs_hvh(self,h_state):
        "hidden to visible"
        v = self.propdown(h_state)
        v_state = v #np.random.binomial(size=v.shape,n=1,p=v)
        
        "then visible to hidden"
        h = self.propup(v_state)
        h_state = h #np.random.binomial(size=h.shape,n=1,p=h)
        
        return v_state,h_state
        
    
    def fit(self,x,k=1,print_loss=True,print_step=1000):
        if x.shape[1] != self.num_input_dim:
            raise ValueError('>>input data dimention not equal to n_input_dim')
        
        self.size = float(x.shape[0])
        
        for i in range(self.num_epoch):
            self.model = {'v':self.v,'hslop':self.hslop,'hbias':self.hbias,'vbias':self.vbias,'vslop':self.vslop}
            
            h = self.propup(x)
            h_state = h #np.random.binomial(size=h.shape,n=1,p=h)
            
            "CD-k. machine dreaming..."
            for step in range(k):
                if step == 0:
                    nx_samples,nh_samples = self.gibbs_hvh(h_state)
                else:
                    nx_means, nh_samples = self.gibbs_hvh(nh_samples)
            
            self.v += self.weight_rate * (x.T.dot(h) - nx_samples.T.dot(nh_samples))/self.size
            self.hbias += self.weight_rate * np.mean(h - nh_samples,axis=0)
            self.vbias += self.weight_rate * np.mean(x - nx_samples,axis=0)
            self.hslop += self.slop_rate * np.mean(h**2 - nh_samples**2,axis=0)/(self.hslop**2)
            #self.vslop += self.slop_rate * np.mean(x**2 - nx_samples**2,axis=0)/(self.vslop**2)
            
            if print_loss and i%print_step == 0:
                monitoring_cost = np.mean((x - nx_samples)**2)
                print(">>Loss after iteration %i: %f"%(i, monitoring_cost))
                
    def reconstruct(self,x,k=1):
        reconstructed_x = x
        for i in range(k):
            h = self.propup(reconstructed_x)
            reconstructed_x = self.propdown(h)
        return reconstructed_x