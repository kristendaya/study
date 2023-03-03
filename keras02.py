#23.02.28

#1.data
#numpy = 수치화 활용 / 데이터를 넣기 위한 배열 =numpy 즉. 데이터를 생성.
import numpy as np
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])
# loss = 0에 수렴 = 최소의loss / 이 1차 함수를회귀모델이라고 함.

#2.model
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

#3.compile
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)
#fit = 훈련을 시키다.
#epochs = 반복 횟수   
#loss = 0.0058

"""
딥러닝은 우리가 빅데이터 등으로 준비한 x(입력값)과 y값(결과값) 을 가지고 컴퓨터에 훈련을 시켜서 
w값과 b값(절편)을 구하는 행위의 반복!!!
이때 컴터는 한가지값을 더 제공하게되는데, 이것이 cost.
COST값은 낮을수록 좋음. 정확한값이 예측하기 위해 ACC 와 PREDIT 사용

X,Y 데이터를 갖추었고, 넘파이를 IMPORT함. 케라스 = 언어가 선행
from keras.models import sequential 
x,y 데이터 준됐고, 이를 사용할수있는 넘파이를 임포트했고, 케라스를 사용할수 이는 환경구축 
model = Sequential() 의미 모델을 순차적으로 구성하겠다는 뜻.
딥러닝을 한다면 데이터를 ㅜㄴ비할때 xy값을 준비하고 얼마나 만흥ㄴ 레이어와 노드를 준비할것인지에 대해 설계해야함.

 epoch는 전체 트레이닝 셋이 신경망을 통과한 횟수 의미

"""