import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

#Data
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])

#[검색] train 과 test를 섞어서 7:3 으로 찾을수 있는 방법!
#힌트 사이킷런
"""
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
print(x_train)
print(x_test)
print(y_train)
print(y_test)


[ 1  8  3 10  5  4  7]
[9 2 6]
[10  3  8  1  6  7  4]
[2 9 5]

"""

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    #train_size=0.7,
    test_size=0.3, 
    random_state=1234, 
    shuffle=True, 
)
print(x_train)
print(x_test)

#모델구성
model=sequential()
model.add(Dense(1,input_dim=1))

#3.컴파일
model.compile(loss='mse',opitmizer='adam')
model.fit(x_train,y_train,epoches=100,batch_size=1)

#평가
loss=model.evaluate(x_test,y_test)
print('loss:',loss)
result=model.predit([11])
print('[11]predict:',result)

#랜덤값을 1-10 뽑아냄. 훈련할때마다 데이터 값이 바뀐다면, 잘만들었는지 못만들었는지 비교가 안됨. 07_mlp1. 참고 
#그러므로 데이터값이 고정되야함. 데이터가 쓰레기면 모델 잘만들어봤자 결과가 안좋음. 랜덤으로 뽑았을지라도 값이 고정. 그게 랜덤시드!
# #[2 1 9 5 6 7 4]
# [ 8  3 10]
