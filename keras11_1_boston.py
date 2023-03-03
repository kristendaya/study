#보스턴에 있는 집값을 맞추는 데이터셋
from sklearn.datasets import load_boston

#data
datasets=load_boston()
x =datasets.data
y =datasets.target

# print(x)
# print(y)

#[6.xxx]전처리를 한수치->첫번째 가격(24.) 

# print(datasets)
# 'feature_names': array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', ->인종열

# print(datasets.feature_names) ->열이 13개 import 13 ...모델링가능

#  print(datasets.DESCR)
# # / instances :506 -> 예시 506개가 있다

print(x.shape,y.shape) #(506, 13) (506,) 스칼라 506, 벡터 1

#[실습]
#1. train 0.7
#2. R2 0.8 이상

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               train_size=0.7,shuffle=True,random_state=1234)
                                              

#model
model=Sequential()
model.add(Dense(1,input_dim=13))
model.add(Dense(10))
model.add(Dense(40))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss="mse",optimizer="adam")
model.fit(x_train,y_train,epochs=200, batch_size=1) #혼자 실험=batchsize 높여보니까 확실히 점들이 퍼짐.
 
#평가,예측
loss=model.evaluate(x_test,y_test)
print("loss:",loss)
y_predict=model.predict(x_test) #훈련안시킨애

#R2,결정계수 =coefficient of determination

from sklearn.metrics import r2_score
r2= r2_score(y_test,y_predict)
print('r2스코어:', r2)