#그림그리자!

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#데이터

x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y=np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               train_size=0.7,shuffle=True,random_state=1234)

#x,y 순서대로 x_train,test,y_train,y_test 계산함 (두번 두번씩)/첫번째가 07비율대로 나누고, 두번째가 또 0.7배율대로 나눠서 

#MODEL 구성
model=Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss="mse",optimizer="adam")
model.fit(x_train,y_train,epochs=30, batch_size=1) #혼자 실험=batchsize 높여보니까 확실히 점들이 퍼짐.
 
#평가,예측
loss=model.evaluate(x_test,y_test)
print("loss:",loss)
y_predict=model.predict(x)

#시각화
import matplotlib.pyplot as plt #뭔지모르나 그림그리는거 댕겨왔다

plt.scatter(x,y)
plt.plot(x,y_predict,color='pink')
#plt.scatter(x,y_predict)
plt.show()
