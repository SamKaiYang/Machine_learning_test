import matplotlib.pyplot as plt
import numpy as np
import random as random
from numpy.random import RandomState
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#part1 #y=ax+b ,a is slope b is intercept
rng = RandomState(2) #seed 影響產出值
x = 10 * rng.rand(50)
y = 3 * x - 5 + rng.randn(50)
plt.scatter(x, y) #繪出二維圖
plt.show() #顯示plt


#part2
from sklearn.linear_model import LinearRegression 

model = LinearRegression(fit_intercept=True) #是否計算該模型的截距

model.fit(x[:, np.newaxis], y) #fit linear model #newaxis表示增加一个新的坐标轴

xfit = np.linspace(0, 10, 1000)#指定生成大小為1000，從0到10的等差數列
yfit = model.predict(xfit[:, np.newaxis]) #predict using the linear model

plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()

print("Model slope:    ", model.coef_[0])
print("Model intercept:", model.intercept_)

#part3 Multidimensional linear models

rng = RandomState(1)
X = 10 * rng.rand(100, 3)
y = 0.5 + np.dot(X, [1.5, -1., 2.])

model.fit(X, y)

print("Multidimensional Model slope:    ",model.coef_)
print("Multidimensional Model intercept:", model.intercept_)

from sklearn.preprocessing import PolynomialFeatures

# x = np.array([2, 3, 4])
# poly = PolynomialFeatures(3, include_bias=False) #來做多項式函數處理
# poly.fit_transform(x[:, None])
# #array([[2.,4.,8.],
# #       [3.,9.,27.],
# #       [4.,16.,64.]]
# # )

from sklearn.pipeline import make_pipeline

poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression()) #一維陣列轉換為三維陣列，加入線性回歸中

rng = RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)

poly_model.fit(x[:, np.newaxis], y)
yfit = poly_model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()