#보스턴에 있는 집값을 맞추는 데이터셋
from sklearn.datasets import load_boston

#data
datasets=load_boston()
x =datasets.data
y =datasets.target

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               train_size=0.7,shuffle=True,random_state=36)
                                              
#(506, 13) (506,) 2번째-스칼라 506, 벡터 1

#model
model=Sequential()
model.add(Dense(2,input_dim=13))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss="mse",optimizer="adam")
model.fit(x_train,y_train,epochs=10, batch_size=2,verbose='auto')

# #b=354 / 1->177/ 1ms/step- 1mm/sec  ==== ->딜레이 ->시간낭비
# #verbose->훈련시키고 보여주고 /보기싫음->  그래서 씀 훈련과정이 없어짐  0->꺼버려라

# loss=model.evaluate(x_test,y_test,verbose=0)
# print("loss:",loss)
# 2->진행바가 안보임. 어느정도 타협점 
#3 ->epoch 만 나옴 ... 3이상은 똑같다. 

###verbose
#0 아무것도안나옴
#1 다 보여줌
#2 프로그래바만 없어져
#위를 제외한 나머지는 에포만 나온다.
#'auto'/ Keras, tensorflow WEB참고 
#dacon/keggle 


y_predict=model.predict(x_test) 


from sklearn.metrics import r2_score
r2= r2_score(y_test,y_predict)
print('r2스코어:', r2)

#RMSE mse는 제곱이잖아 r루트 씌운거. 요즘 R 안씀 /빅데이터분석기사 