import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import math

def get_filters(count,m,n,p):
	mini=-1
	maxi=1
	fil=[]
	for i in range(0,count):
		fil.append(( np.random.rand(m,n,p) * (maxi - mini) ) + mini)
	fil=np.array(fil)
	return fil

def gaussian_kernel(size, size_y=None):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size:size+1, -size_y:size_y+1]
    g = np.exp(-(x**2/float(size)+y**2/float(size_y)))
    return g / g.sum()

def convolution(m,f):
#stride 1
	final=np.zeros((len(m)-len(f)+1,len(m[0])-len(f[0])+1),dtype=float)
	l1=len(f)
	l2=len(f[0])
	for x in range(0,len(m)-l1+1):
		for y in range(0,len(m[0])-l2+1):
			for p in range(x,x+l1):
				for q in range(y,y+l2):
					for z in range(0,len(f[0][0])):
						final[x][y]+=m[p][q][z]*f[p-x][q-y][z]
	return final

def pooling(Matrix,m,n,stride):
	pooled_output=[]
	for p in range(0,len(Matrix)):
		M=Matrix[p]
		pooled=np.zeros((len(M)/2,len(M[0])/2),dtype=float)
		for i in range(0,len(M)-m+1,stride):
			for j in range(0,len(M[0])-n+1,stride):
				maxi=-100000
				for x in range(i,i+m):
					for y in range(j,j+n):
						maxi=max(maxi,M[x][y])
				pooled[i/2][j/2]=maxi
		pooled_output.append(pooled)
	return np.array(pooled_output)

def initialise_weights(m,n):
    maxi=0.1
    mini=0.9
    w= ( np.random.rand(m,n) * (maxi - mini) ) + mini
    return w

def tanh(val):
    ret_val = math.tanh(val)
    return ret_val
def Relu(val):
	return max(0,val)

im =img.imread("1.png")

#layer1
fil_1=get_filters(6,5,5,3)
conv1_output=[]
for i in range(0,6):
	conv1_output.append(convolution(im,fil_1[i]))
conv1_output=np.array(conv1_output)
# for i in range(0,6):
# 	plt.imsave("1-"+str(i),conv1_output[i])

#layer2
pooled1_output=pooling(conv1_output,2,2,2)
pooled1_output=np.rollaxis(pooled1_output,0,3)

#layer3
conv2_output=[]
fil_2=get_filters(16,5,5,6)
for i in range(0,16):
	conv2_output.append(convolution(pooled1_output,fil_2[i]))
conv2_output=np.array(conv2_output)

#layer4
pooled2_output=pooling(conv2_output,2,2,2)

values=[]
for i in range(0,16):
	for j in range(0,5):
		for k in range(0,5):
			values.append(pooled2_output[i][j][k])

print values

values.append(1)
#layer5
w_1=initialise_weights(120,len(values))
input_layer5=np.matmul(w_1,values)
print input_layer5
output_layer5=np.zeros(len(input_layer5)+1,dtype=float)
for i in range(0,len(input_layer5)):
	output_layer5[i]=Relu(input_layer5[i])

print output_layer5
output_layer5[len(output_layer5)-1]=1

#layer6
w_2=initialise_weights(84,len(output_layer5))
input_layer6=np.matmul(w_2,output_layer5)
print input_layer6
output_layer6=np.zeros(len(input_layer6)+1,dtype=float)
for i in range(0,len(input_layer6)):
	output_layer6[i]=Relu(input_layer6[i])
print output_layer6
output_layer6[len(output_layer6)-1]=1
#layer7
g=gaussian_kernel(5,len(output_layer6)/2)
output_layer7=np.matmul(g,output_layer6)
print output_layer7
