#강제로 나쁘게 만들어라
#아주 나쁘면 음수가 나옴. R2를 음수가 아닌 0.5 이하로 만들것
#2.데이터는 건들지말것
#3.레이어는 인풋,아웃품 포함 7이상
#4.batch_size=1(고정)
#5.히든레이어의 노드는 10개이상 100개 이하(=일단위가 있으면 안됨)
#6.train 사이즈 75%
#7.epoch 100번이상
#8.loss 지포는 mse,mae 
#[실습시작]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#데이터

x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y=np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               train_size=0.75,shuffle=True,random_state=50)

#x,y 순서대로 x_train,test,y_train,y_test 계산함 (두번 두번씩)/첫번째가 07비율대로 나누고, 두번째가 또 0.7배율대로 나눠서 

#MODEL 구성
model=Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(11))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss="mse",optimizer="adam")
model.fit(x_train,y_train,epochs=100, batch_size=1) #혼자 실험=batchsize 높여보니까 확실히 점들이 퍼짐.
 
#평가,예측
loss=model.evaluate(x_test,y_test)
print("loss:",loss)
y_predict=model.predict(x_test) #훈련안시킨애

#R2,결정계수 =coefficient of determination

from sklearn.metrics import r2_score
r2= r2_score(y_test,y_predict)
print('r2스코어:', r2)