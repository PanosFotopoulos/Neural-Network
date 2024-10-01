import math

softmax_output = [0.7, 0.1, 0.2]

#target class = 0, that means the index 0 of softmax output is the "hot" or equals to one

target_output = [1, 0, 0] # target destribution.

loss1 = -(math.log(softmax_output[0])*(target_output[0]) +
         math.log(softmax_output[1])*(target_output[1]) +
         math.log(softmax_output[2])*(target_output[2])
         )
        
print (f'My loss is: {loss1}')

loss2 = -math.log(softmax_output[0])*target_output[0]

print (f'My loss is: {loss2}')

if loss1==loss2:
    print ('The losses are Equal')
    
'''
This is the resson Log is very important. As you can notice (softmax_output[1])*(target_output[1]) and (softmax_output[2])*(target_output[2]) have output
equal to 0
'''