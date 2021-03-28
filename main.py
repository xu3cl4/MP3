import cupy
import cupy as np
from tensorflow.keras.datasets import mnist
from tqdm import tqdm_notebook as tqdm
from functools import reduce

def evaluate_acc(y, y_pred):
    y = np.argmax(y, axis=-1)
    y_pred = np.argmax(y, axis=-1)
    match = y[y==y_pred]
    print(match.shape[0]/y.shape[0])

GAMMA = 0.2
# for ReLU
def He(input_nodes, output_nodes):
    return np.random.randn(output_nodes, input_nodes)*np.sqrt(2/input_nodes) 

# for tanh
def Xavier(input_nodes, output_nodes):
    return np.random.randn(output_nodes, input_nodes)*np.sqrt( 2/(input_nodes+output_nodes) )

# else
def random(*shape):
    total_samples = reduce(lambda x,y: x * y,shape,1)
    return np.random.randn(total_samples).reshape((shape)) * 0.01

def logistic(x):
    return 1./(1.+np.exp(-x))

def dlogistic(x):
    log = logistic(x)
    return log * (1 - log)


#hyperbolic tangent
def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1/(np.cosh(x)**2)


#Rectified Linear Unit ReLU
def ReLU(x):
    return np.maximum(0,x)

def dReLU(x):
    y = np.copy(x)
    y[y>=0] = 1
    y[y<0] = 0
    return y


#leaky ReLU
def lReLU(x, gamma=GAMMA):
    return np.maximum(0,x) + gamma * np.minimum(0,x)

def dlReLu(x, gamma=GAMMA):
    y = np.copy(x)
    y[y>=0] = 1
    y[y<0] = gamma
    return y


#softplus
def softplus(x):
    return np.log(1.+np.exp(x))

def dsoftplus(x): #same as logistic
    return 1./(1.+np.exp(-x))


# softmax
def softmax(x):     # stable softmax function
    exp = np.exp(x - np.max(x))
    return exp/np.sum(exp, axis=1,keepdims=True)

def dsoftmax(x):
    '''
    this function uses the einstein summation
    it leverages the fact that instead of doing -s[i] * s[j] for i != j and s[i] * (1 - s[i]) for i = j
    it instead becomes -s[i] * s[j] for i != j and s[i] - s[i] ^2  for i = j
    we can separate it as 0 - s[i]s[j] for i !=j and s[i] - s[i]s[i] for i = j
    We can thus initialize a matrix which diagonal entries are s[i] and substract a matrix whose values are s[i]s[j]
    Withhout handling batch, it becomes np.diagflat(s) - np.dot(s,s.T)
    With batch, we really need einstein summation. Copied from https://stackoverflow.com/questions/33541930/how-to-implement-the-softmax-derivative-independently-from-any-loss-function 
    '''
    s = softmax(x)
    flat_diag = np.einsum('ij,jk->ijk',s,np.eye(s.shape[-1])) # creates a diag from softmax entries for each batch (batch_size, DIAGONAL)
    sub = np.einsum('ij,ik->ijk', s, s) #  do the -s[i] * s[j] along each batch for each elements
    return flat_diag - sub

def cross_entropy(y, y_hat): 
    return -np.sum(y*np.log(y_hat), axis=1,keepdims=True)

def dcross_entropy(y, y_hat):
    return -y/y_hat

def cross_entropy_L2(y, y_hat, weights, lambd):
    pass

def dcross_entropy_L2(y, y_hat, weights, lambd):
    pass

def mse(y,y_hat):
    return 0.5 * np.mean((y - y_hat) **2)
def dmse(y,y_hat):
    return y - y_hat


def transform_x(xs):
    mean = np.mean(xs)
    std = np.std(xs)
    res =  (xs - mean)/std
    return res.reshape((res.shape[0],784))
def transform_y(ys,num_classes=10):
    ohs = np.zeros((ys.shape[0],num_classes))
    ohs[np.arange(0,ys.shape[0],1),ys] = 1
    return ohs

def load_data():
    (X_train , Y_train), (X_test, Y_test) = mnist.load_data() 
    X_train = cupy.asarray(transform_x(X_train))
    Y_train = cupy.asarray(transform_y(Y_train))
    X_test = cupy.asarray(transform_x(X_test))
    Y_test = cupy.asarray(transform_y(Y_test))
    return (X_train, Y_train), (X_test, Y_test)



INITIALIZERS = {
        "he" : He,
        "xavier" : Xavier,
        "random" : random
        }

ACTIVATIONS = {
        "relu" : [ReLU,dReLU],
        "sigmoid" : [logistic,dlogistic],
        "tanh" : [tanh,dtanh],
        "lrelu" : [lReLU,dlReLu],
        "softplus" : [softplus,dsoftplus],
        "softmax" : [softmax,dsoftmax],
        }

LOSSES = {
        "crossentropy" : [cross_entropy,dcross_entropy],
        "mse": [mse,dmse],
        "crossentropy_L2" : [cross_entropy_L2, dcross_entropy_L2]
}


# could add to base class in the future maybe
class Layer:pass

class Optimizer:pass

class SGDOptimizer:
    def __init__(self,lr):
        self.lr = lr
    def __call__(self,gradient):
        deltas = -self.lr * np.mean(gradient,axis=0)
        return deltas

class Dense(Layer):
    def __init__(self,nodes,initializer=INITIALIZERS["he"],bias_initializer=INITIALIZERS["random"]):
        self.initializer = initializer
        self.nodes = nodes
        self.bias_initializer = bias_initializer
    def set_optimizer(self,optimizer):
        self.optimizer = optimizer
    def initialize(self,input_nodes):
        output_nodes = self.nodes
        self.weights = self.initializer(input_nodes,output_nodes)
        self.biases = self.bias_initializer(output_nodes)
        self.input_nodes = input_nodes
        self.output_nodes = self.nodes
        return self.nodes
    def summary(self):
        return f'Dense Layer: input={self.input_nodes},output={self.output_nodes}'
    def forward(self,inputs):
        self.inputs = inputs
        return self.inputs @ self.weights.T + self.biases # believe me it works

    def backward(self,delc_delz):
        delc_delb = delc_delz
        delc_delw = np.einsum('ij,ik -> ikj',self.inputs,delc_delz)
    
        delc_delx = delc_delz @ self.weights
        self.weights += self.optimizer(delc_delw)
        self.biases += self.optimizer(delc_delb)

        return delc_delx

class Activation(Layer):
    def __init__(self,act_name):
        self.act_name = act_name
    def forward(self,inputs):
        assert self.act_name is not None or self.act_name != "", "Activation function must be defined"
        self.inputs = inputs
        return ACTIVATIONS[self.act_name][0](self.inputs)
    def summary(self):
        return f"Activation: {self.act_name}"
    def backward(self,delc_dela):
        '''this should work for all R -> R activation functions. However softmax being R(n) -> R(n) must be overwritten'''
        assert self.act_name is not None or self.act_name != "", "Activation function must be defined"
        dela_delz = ACTIVATIONS[self.act_name][1](self.inputs)
        dim_size = len(dela_delz.shape)
        assert dim_size == 2 or dim_size == 3
        if dim_size == 2:
            return delc_dela * dela_delz
        else:
            return np.einsum('ij,ijk -> ik',delc_dela,dela_delz) # trust me it works

class Loss(Layer):
    def __init__(self,loss_name):
        self.loss_name = loss_name
    def forward(self,preds,targets):
        return LOSSES[self.loss_name][0](preds,targets)
    def backward(self,preds,targets):
        return LOSSES[self.loss_name][1](preds,targets)

class Model:
    def __init__(self,layers,input_size):
        self.layers = layers
        self.input_size = input_size
        self.compiled = False
    def compile(self,loss,optimizer,metrics=[]):
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.compiled = True
        self.init_dense()
    def summary(self):
        result = '\n---------------------\n'.join([layer.summary() for layer in self.layers])
        print(result)
    def forward(self,inputs):
        return reduce(lambda xs,layer : layer.forward(xs),self.layers,inputs)
    
    def init_dense(self):
        '''intializes all dense layers with the correct input-output nodes count'''
        prev_nodes = self.input_size
        for layer in self.layers:
            if isinstance(layer,Dense):
                layer.set_optimizer(self.optimizer)
                prev_nodes = layer.initialize(prev_nodes)


    def backward(self,preds,targets,count):
        loss_val = self.loss.forward(preds,targets)
        #if count % 10 == 0:
        #    print(f"loss {np.mean(loss_val).tolist():.3f}")
        reduce(lambda delc, layer : layer.backward(delc),reversed(self.layers),self.loss.backward(preds,targets))
        return loss_val

    def predict(self,inputs):
        return self.forward(inputs)

    def fit_sample(self, inputs, outputs, count=1, x=None, y=None, track_iter=False):
        assert self.compiled, "Please compile the model first, with model.compile(loss,optimizer,metrics)"
        preds = self.forward(inputs)
        self.backward(preds, outputs,count)

        # calculate testing accuracy to determine max_iter
        if track_iter and count % 10 == 0:
           y_preds = self.forward(x)
           print(f"At the {count} iteration, training accuracy is {evaluate_acc(y_true, y_preds)}")

    def fit(self,inputs,outputs,batch_size=16,epochs=1,count=1, max_iter=10000, track_iter=False):
        num_batches = int(inputs.shape[0] / batch_size)
        for epoch in range(epochs):
            for batch_i in tqdm(range(num_batches),desc=f"epoch {epoch} of {epochs}"):
                if count > max_iter: break 
                batch_start = batch_i * batch_size
                batch_end = batch_i * batch_size + batch_size
                batch_in = inputs[batch_start:batch_end]
                batch_out = outputs[batch_start:batch_end]
                if track_iter:
                   self.fit_sample(batch_in, batch_out, count=count, x=inputs, y=outputs, track_iter=track_iter) 
                else:
                   self.fit_sample(batch_in, batch_out, count=count)
                count += 1

if __name__ == "__main__":
    model = Model([
        Dense(124),
        Activation("relu"),
        Dense(124),
        Activation("relu"),
        Dense(10),
        Activation("softmax")

    ],784)
    # model = Model([
        # Dense(2),
        # Activation("sigmoid"),
        # Dense(2),
        # Activation("softmax")
    # ],2)
    LR = 3e-1
    model.compile(Loss("crossentropy"),SGDOptimizer(LR))
    model.summary()

    (X_train, Y_train), (X_test, Y_test) = load_data()
    model.fit(X_train,Y_train,batch_size=16,epochs=1)
    # xor problem
    # inputs = np.array([
        # [1,1],
        # [1,0],
        # [0,1],
        # [0,0]
    # ])
    # outputs = np.array([
        # [0],
        # [1],
        # [1],
        # [0]
    # ])
    # outputs = np.array([
        # [1,0],
        # [0,1],
        # [0,1],
        # [1,0]
    # ])
    # model.fit(inputs,outputs,epochs=10000)
    # print(model.predict(inputs))
