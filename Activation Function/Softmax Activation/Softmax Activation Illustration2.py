#Since we understand well the basics. Now we gona convert this to numpy 

import numpy as np

'''
Converting to numpy and applying Exponentiate to the data
'''

#import math
#layer_output = [[4.8, 1.21, 2.385],
#E = 2.171828182846
#E= math.e
#exp_values = []

#for output in layer_output:
#    exp_values.append(E**output)    
#print (f'This is euler number in power of layer output values  {exp_values}')

layer_output = [[4.8, 1.21, 2.385],
                [8.9, -1.81, 0.2],
                [1.41, 1.051, 0.026]]


exp_values = np.exp(layer_output)  #Exponentiate values
print (f'Euler number in power of my values are :  {exp_values}')

'''
Now we need to create a vector of 3 values (and in future reshape it) like before
but if we print print(np.sum(layer_output)) we get 1 values 
'''


#norm_values = exp_values/ np.sum(exp_values)
    
#print(f'The probability distribution: {norm_values}')#The normalized values of exp_values are
#print(f'The sum of those are very close to 1: {sum(norm_values)}')

'''
sum(layer_output) axis=1
layer_output = [[4.8, 1.21, 2.385], = 8.395  
                [8.9, -1.81, 0.2], = 7.29     
                [1.41, 1.051, 0.026]] = 2.487  
'''

print(f'Changing dimention to the vector {np.sum(layer_output,axis=1, keepdims=True)}')

norm_values = exp_values/np.sum(exp_values, axis= 1, keepdims=True)

'''
e**1/(e**1 + e**2 + e**3),  e**2/(e**1 + e**2 + e**3),  e**3/(e**1 + e**2 + e**3)
e**4/(e**1 + e**2 + e**3),  e**5/(e**1 + e**2 + e**3),  e**6/(e**1 + e**2 + e**3)
e**6/(e**1 + e**2 + e**3),  e**8/(e**1 + e**2 + e**3),  e**9/(e**1 + e**2 + e**3)
'''

print(f'Normalized values are: {norm_values}')
