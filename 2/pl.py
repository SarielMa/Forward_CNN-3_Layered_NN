import matplotlib.pyplot as pl

#tanh
x1=[6,8,10,12,14,16,18,20,22,24,26]
y1=[0.995867768595,1,0.99173553719,1,0.995867768595,0.995867768595,0.99173553719,0.995867768595,1,1,1]
y2=[0.929752066116,0.99173553719,0.99173553719,0.99173553719,0.995867768595,1.0,0.99173553719,0.987603305785,0.99173553719,1,0.995867768595]
y3=[0.995867768595,0.995867768595,1,1,1,1,1,0.995867768595,0.995867768595,1,0.995867768595]
pl.xlabel("No of hidden layers")
pl.ylabel("5-fold cross validation accuracy")
pl.title("dermatology dataset")
#pl.plot(x1,y1,'r')

#pl.plot(x2,y2,'blue', label='ionosphere voted_perceptron')

#pl.show()


pl.figure(1)
pl.plot(x1, y1, 'blue', label='tanh')
pl.plot(x1, y2, 'red', label='Relu')
pl.plot(x1, y3, 'orange', label='sigmoid')

#plt.figure(2)
#plt.plot(epochs, cancer_acc_voted, 'green', label='cancer voted_perceptron')
#plt.plot(epochs, cancer_acc_normal, 'red', label='cancer perceptron')
#plt.legend()
pl.show()
