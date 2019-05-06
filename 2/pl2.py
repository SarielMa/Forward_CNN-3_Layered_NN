import matplotlib.pyplot as pl

#tanh
x1=[6,8,10,12,14,16,18,20,22,24,26]
y1=[0.991596638655,0.995098039216,0.995798319328,0.992296918768,0.993697478992,0.994397759104,0.995798319328,0.987394957983,0.993697478992,0.994397759104,0.992296918768]
x2=[26,24,22,20,18,16,14,12,10,8,6]
y2=[0.990896358543,0.988795518207,0.99299719888,0.991596638655,0.994397759104,0.990196078431,0.99299719888,0.992296918768,0.99299719888,0.974089635854,0.91106442577]
y3=[0.994397759104,0.993697478992,0.985994397759,0.995798319328,0.995798319328,0.981792717087,0.993697478992,0.994397759104,0.993697478992,0.991596638655,0.995098039216]
pl.xlabel("No of hidden layers")
pl.ylabel("5-fold cross validation accuracy")
pl.title("pendigits dataset")
#pl.plot(x1,y1,'r')

#pl.plot(x2,y2,'blue', label='ionosphere voted_perceptron')

#pl.show()


pl.figure(1)
pl.plot(x1, y1, 'blue', label='tanh')
pl.plot(x2, y2, 'red', label='Relu')
pl.plot(x2, y3, 'orange', label='sigmoid')
pl.legend()
#plt.figure(2)
#plt.plot(epochs, cancer_acc_voted, 'green', label='cancer voted_perceptron')
#plt.plot(epochs, cancer_acc_normal, 'red', label='cancer perceptron')
#plt.legend()
pl.show()
