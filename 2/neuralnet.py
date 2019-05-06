import numpy as np
import math
import random

def data_preprocessing(filename,values):
    contents = np.genfromtxt(filename, delimiter=',',missing_values='?',usemask=False)
    contents=contents[~np.isnan(contents).any(axis=1)]
    contents=contents.astype(float)
    np.random.shuffle(contents)
    attributes_num=len(contents[0])
    x_data=[]
    y_data=[]
    for i in range(0,len(contents)):
        y=contents[i][attributes_num-1]
        if y in values:
            y_data.append(contents[i][attributes_num-1])
            x_data.append(list(contents[i][0:attributes_num-1]))
    x_data=np.array(x_data).astype(float)
    y_data=np.array(y_data).astype(int)
    deviation=np.std(x_data,axis=0)
    column_mean=np.mean(x_data,axis=0)
    for i in range(0,len(x_data)):
        x_data[i]=np.divide((x_data[i]-column_mean),deviation)
    for i in range(0,len(x_data)):
        x_data[i]=x_data[i]+[1.0]
    return [x_data,y_data]

def initialise_weights(dim_x,hidden_layer_nodes,output_layer_nodes):
    maxi=0.5
    mini=-0.5
    w_h = ( np.random.rand(hidden_layer_nodes,dim_x) * (maxi - mini) ) + mini
    w_o = ( np.random.rand(output_layer_nodes,hidden_layer_nodes+1) * (maxi - mini) ) + mini
    return [w_h,w_o]

def sigmoid(val):
    ans=(1.0+math.exp(-val*1.0))
    ans=1.0/ans
    return ans

def tanh(val):
    ret_val = math.tanh(val)
    return ret_val

def Relu(val):
    return max(0.0,val*1.0)

def softmax(l):
    l1=np.exp(l-max(l))
    return l1/l1.sum()

def derivative_softmax(l,index):
    l=softmax(l)
    return l[index]*(1-l[index])

def derivative_Relu(val):
    if val>0:
        return 1
    else:
        return 0
def derivative_tanh(val):
    val=val*1.0
    ret_val1 = math.tanh(val)
    ret_val = 1-(ret_val1*ret_val1)
    return ret_val

def derivative_sigmoid(val):
    ret_val1 = 1 + math.exp(-1*val)
    ret_val1 = 1/ret_val1
    ret_val = ret_val1*(1-ret_val1)
    return ret_val

def forward_propagation(x,w_h,w_o):
    net_h=np.ones(hidden_layer_nodes,dtype=float)
    net_o=np.ones(output_layer_nodes,dtype=float)
    y_h=np.ones(hidden_layer_nodes+1,dtype=float)
    z_o=np.ones(output_layer_nodes,dtype=float)

    for j in range(0,len(net_h)):
        net_h[j]=np.dot(w_h[j],x)
        y_h[j]=sigmoid(net_h[j])

    for k in range(0,len(net_o)):
        net_o[k]=np.dot(w_o[k],y_h)
        # z_o[k]=sigmoid(net_o[k])
    z_o=softmax(net_o)

    return [net_h,net_o,y_h,z_o]

def backward_propagation(net_h,net_o,y_h,z_o,output_layer_nodes,y_val):
    delta_o=np.zeros(output_layer_nodes,dtype=float)
    for n in range(0,len(z_o)):
        if n==int(y_val)-1:
            delta_o[n]=(1-z_o[n])
        else:
            delta_o[n]=(-1.0*z_o[n])
    delta_h=np.zeros(hidden_layer_nodes+1,dtype=float)
    for j in range(0,len(net_h)):
        for n in range(0,output_layer_nodes):
            delta_h[j]+=derivative_sigmoid(net_h[j])*delta_o[n]*w_o[n][j]

    return [delta_h,delta_o]

def neuralnet(dim_x,hidden_layer_nodes,output_layer_nodes,x_data,w_h,w_o,y_data):
    while 1:
        delta_w_h=np.zeros((hidden_layer_nodes,dim_x),dtype=float)
        delta_w_o=np.zeros((output_layer_nodes,hidden_layer_nodes),dtype=float)
        eta=0.1
        total=0
        for l in range(0,len(x_data)):
            [net_h,net_o,y_h,z_o]=forward_propagation(x_data[l],w_h,w_o)
            [delta_h,delta_o]=backward_propagation(net_h,net_o,y_h,z_o,output_layer_nodes,y_data[l])

            for j in range(0,len(delta_w_h)):
                for i in range(0,len(delta_w_h[0])):
                    w_h[j][i]=w_h[j][i]+eta*delta_h[j]*x_data[l][i]
            for n in range(0,len(delta_w_o)):
                for j in range(0,len(delta_w_o[0])):
                    w_o[n][j]=w_o[n][j]+eta*delta_o[n]*y_h[j]
                    total+=(eta*delta_o[n]*y_h[j])**2
        total=math.sqrt(total)
        # print total
        if total<=0.7:
            break
    return [w_h,w_o]

values=[1,2,3,4]
# [x_data,y_data]=data_preprocessing('dermatology.data',values)
[x_data,y_data]=data_preprocessing('pendigits.tes',values)
hidden_layer_nodes=8
output_layer_nodes=len(values)
dim_x=len(x_data[0])
l=len(x_data)
k=5
p=l/k
final=0
c=0
for ind in range(1,k+1):
    [w_h,w_o]=initialise_weights(dim_x,hidden_layer_nodes,output_layer_nodes)
    #test_data -> [(ind-1)*p,ind*p)
    if ind<k:
        end=ind*p
    else:
        end=l

    test_data_x=x_data[(ind-1)*p:end]
    test_data_y=y_data[(ind-1)*p:end]

    train_data_x=np.append(x_data[0:(ind-1)*p],x_data[end:l],axis=0)
    train_data_y=np.append(y_data[0:(ind-1)*p],y_data[end:l],axis=0)

    [w_h,w_o]=neuralnet(dim_x,hidden_layer_nodes,output_layer_nodes,train_data_x,w_h,w_o,train_data_y)
    ans=0
    for i in range(0,len(test_data_x)):

        [net_h,net_o,y_h,z_o]=forward_propagation(test_data_x[i],w_h,w_o)
        maxi=max(z_o)
        if z_o[int(test_data_y[i])-1]==maxi:
            ans+=1
        # print z_o
    #print ans,len(test_data_x)
    final+=ans*1.0
    c+=len(test_data_x)*1.0
print final/c
