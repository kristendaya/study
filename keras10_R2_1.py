#그림그리자!

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#데이터

x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y=np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               train_size=0.9,shuffle=True,random_state=1234)

#x,y 순서대로 x_train,test,y_train,y_test 계산함 (두번 두번씩)/첫번째가 07비율대로 나누고, 두번째가 또 0.7배율대로 나눠서 

#MODEL 구성
model=Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(6))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(6))
model.add(Dense(1))

model.compile(loss="mse",optimizer="adam")
model.fit(x_train,y_train,epochs=500, batch_size=1) #혼자 실험=batchsize 높여보니까 확실히 점들이 퍼짐.
 
#평가,예측
loss=model.evaluate(x_test,y_test)
print("loss:",loss)
y_predict=model.predict(x_test) #훈련안시킨애

#R2,결정계수 =coefficient of determination

from sklearn.metrics import r2_score
r2= r2_score(y_test,y_predict)
print('r2스코어:', r2)

#튜닝해서 0.99로 올려ㅗ기 
#혼자 실험 : 같은값으로 했는데 mae가 더 적게나옴/Dense 를 하나 높이니까 0.1 상승/ EPOCH 500이나 800이나 별차이없음. 최대가 r2=0.78 
#히든레이어를 촘촘하게 짜는게 확률을 높이는것같다(추측)->아니다.
# 100으로 높인이후로 0.8019401617453955
#0.99가 좋다.
#Train_size 0.8로 올리고 r2스코어: 0.8733508209826807
#r2스코어: 0.9284069134633586/0.943234889005137/r2스코어: 0.9621389214030986
#train size변경가능 randomstate 도