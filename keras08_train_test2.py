import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

#Data
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([10,9,8,7,6,5,4,3,2,1,])

#[실습]]넘파이 리스트의 슬라이싱! 7:3으로 잘라라
x_train = x[:7] #[1-7]]
x_test=x[7:10]  
print(x_train)
print(x_test)
y_train=y[:7]
y_test=y[7:]


"""
print(x_train.shape,x_test.shape)  #(7,)(3,)
print(y_train.shape,y_test.shape)  #(3,)(3,)

#(0,) (3,)
#(4,) (2,)

"""
