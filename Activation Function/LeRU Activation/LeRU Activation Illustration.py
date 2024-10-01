
def relu(inputs):
    output = []   
       
    for i in inputs:
        if i>0:
            output.append(i)
        elif i <=0:
            output.append(0)
        
    return output



inputs =  [-1, 1, 0, -2, 2]
result = relu(inputs)
print (result)