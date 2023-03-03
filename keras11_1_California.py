from sklearn.datasets import fetch_california_housing

#1.data

datasets=fetch_california_housing()
x=datasets.data
y=datasets.target

print(x.shape,y.shape) #(20640, 8) (20640,)

#[실습]
# R2 0.55-0.6이상

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               train_size=0.9,shuffle=True,random_state=1234)
                                              

#model
model=Sequential()
model.add(Dense(1,input_dim=8))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss="mse",optimizer="adam")
model.fit(x_train,y_train,epochs=50, batch_size=1)
#혼자 실험=batchsize 높여보니까 확실히 점들이 퍼짐.
 
#평가,예측
loss=model.evaluate(x_test,y_test)
print("loss:",loss)
y_predict=model.predict(x_test) #훈련안시킨애

#R2,결정계수 =coefficient of determination

from sklearn.metrics import r2_score
r2= r2_score(y_test,y_predict)
print('r2스코어:', r2)

# 0.4116824385421938

