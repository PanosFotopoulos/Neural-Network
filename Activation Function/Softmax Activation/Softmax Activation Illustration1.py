import math
layer_output = [4.8, 1.21, 2.385]

#E = 2.171828182846
E= math.e

exp_values = []

for output in layer_output:
    exp_values.append(E**output)
    
print (f'This is euler number in power of layer output values  {exp_values}')

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append( value / norm_base)
    
print(f'The probability distribution: {norm_values}')#The normalized values of exp_values are
print(f'The sum of those are very close to 1: {sum(norm_values)}')

#Move to part 2 in order to implement it in numpy.
