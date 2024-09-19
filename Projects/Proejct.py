import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib as plt

nnfs.init()

np.random.seed(0)

X =  [[1, 2, 3, 2.5],
      [2.0,5.0,-1.0,2.0],
      [-1.5,2.7,3.3,-0.8]]



class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        

class Activation_ReLU: 
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)
        
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs- np.max(inputs, axis=1, keepdims=True))
        probabilties = exp_values/ np.sum(exp_values, axis=1, keepdims= True)
        self.output = probabilties
        
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self,y_pred, y_true):
        sample = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(sample),y_true]
        elif len(y_true.shape) == 2: #onehot encoded vectors
            correct_confidences = np.sum(y_pred_clipped*y_true, axis= 1)
            
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

'''
1st case
sample = 3 
y_pred_clipped = np.array([[0.7,  0.1,  0.2],
                           [0.1,  0.5,  0.4]
                           [0.02, 0.9,  0.08]])
y_true = [0, 1, 1] 

if len(y_true.shape) == 1: #onehot
    correct_confidences = y_pred_clipped[range(sample),y_true]                        
    
    
 y_pred_clipped      [range(sample),        y_true]             
[[0.7,  0.1,  0.2],       [0,                  [0                 0.7
 [0.1,  0.5,  0.4]         1,                   1,          =     0.5
[0.02, 0.9,  0.08]]        2]           ,       1]                0.9


2nd case
y_pred_clipped = np.array([[0.7,  0.1,  0.2],
                           [0.1,  0.5,  0.4]
                           [0.02, 0.9,  0.08]])
y_true = [[1, 0, 0]
          [0, 1, 0]
          [0, 1, 0]]
          
elif len(y_true.shape) == 2:
    correct_confidences = np.sum(y_pred_clipped*y_true, axis= 1)
    
    
 y_pred_clipped                 y_true]             
[[0.7,  0.1,  0.2],           [[1, 0, 0]                 [[0.7, 0.0, 0.0],
[0.1,  0.5,  0.4]      *       [0, 1, 0],          =      [0.5, 0.5, 0,0],     
[0.02, 0.9,  0.08]]            [0, 1, 0]]                 [0.0, 0.9, 0.0]]



 y_pred_clipped                 y_true]             
[[0.7,  0.1,  0.2],           [[1, 0, 0]               np.sum  ([[0.7, 0.0, 0.0],                [0.7,
[0.1,  0.5,  0.4]      *       [0, 1, 0],          =             [0.0, 0.5, 0,0],           =     0.5,
[0.02, 0.9,  0.08]]            [0, 1, 0]]                        [0.0, 0.9, 0.0]], axis= 1)       0.9]
''' 
        

        
        
        
X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
actvation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()


dense1.forward(X)
actvation1.forward(dense1.output)

dense2.forward(actvation1.output)
activation2.forward(dense2.output)

#print(activation2.output[:5])

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)

print('Loss:', loss)