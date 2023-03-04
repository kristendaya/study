from sklearn.datasets import load_diabetes

#1.데이터
datasets = load_diabetes()
x = datasets.data
y=datasets.target

print(x.shape,y.shape) #모양먼저 찍어보면 모양어떻게할지. #(442,10), (442,)

#[실습]
# R2 0.62 이상
##데이터정제를 잘해야함. (아직 실력안됨.)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               train_size=0.9,shuffle=True,random_state=72)
                                              

#model
model=Sequential()
model.add(Dense(1,input_dim=10))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(70))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss="mse",optimizer="adam")
model.fit(x_train,y_train,epochs=3000, batch_size=20) 

#평가,예측
loss=model.evaluate(x_test,y_test)
print("loss:",loss)
y_predict=model.predict(x_test) 

#R2,결정계수 =coefficient of determination

from sklearn.metrics import r2_score
r2= r2_score(y_test,y_predict)
print('r2스코어:', r2)

#R2 = 0.64 
