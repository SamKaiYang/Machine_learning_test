import matplotlib
import matplotlib.pyplot as plt 
import numpy as np 
import random as random
import csv

#create some data
#training data
x_data=[338, 333, 328, 207, 226, 25, 179, 60, 208, 606]

y_data=[640, 633, 619, 393, 428,27, 193, 66, 226, 1591]


x = np.arange(-200,-100,1) #bias #numpy array 等差
y = np.arange(-5,5,0.1) #weight
Z =  np.zeros((len(x), len(y)))
X, Y = np.meshgrid(x, y) #將向量x和y定義的區域轉換成矩陣X和Y,其中矩陣X的行向量是向量x的簡單複制，而矩陣Y的列向量是向量y的簡單複制
for i in range(len(x)): #len 矩陣內參數數量
    for j in range(len(y)):
        b = x[i]
        w = y[j]
        Z[j][i] = 0
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] +  (y_data[n] - b - w*x_data[n])**2
        Z[j][i] = Z[j][i]/len(x_data)

#initial weight and bias
b=-120
w=-4
lr=0.0000001 #須更改
iteration=100000

b_history=[b]
w_history=[w]

for i in range(iteration):
#gradient fxn
    b_grad=0.0   # 新的b點位移預測
    w_grad=0.0   # 新的w點位移預測
    
    for n in range(len(x_data)):
        #p.18
        # L(w,b)對b偏微分
        #b_grad = ???
        # L(w,b)對w偏微分
        #w_grad = ???
        
    #update parameter #AdaGrad


#p.13
    #b = ??? 
    #w = ???

    b_history.append(b)
    w_history.append(w)  #put all the b,w into the array, find the minimum



#plot the figure

 
plt.contourf(x,y,Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))    
plt.plot([-188.4], [2.67], 'x', ms=12, markeredgewidth=3, color='orange')
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()
