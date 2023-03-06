#문제풀이
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
#pandas

#.DATA
path ="./_data/ddareung/" #.->current folder /-> under .->study_main 

train_csv = pd.read_csv(path +"train.csv",
                        index_col= 0 ) #문자+문자 =뭉쳐버림
# train_csv = pd.read_csv(./_data/ddareung/train.csv) 원래는 이렇게 써야되는데 path 쓰는건 자주쓸거같아서 

print(train_csv)
print(train_csv.shape) #(1459,10)

#헤더와 인덱스는 따로 연산하지 않는다.

test_csv = pd.read_csv(path +"train.csv",
                        index_col= 0 )

print(test_csv)
print(test_csv.shape)

#id 는 데이터가 아니라 표시하는거. data 키,몸무게... 18개 .그냥 번호만 매김. 처음에 만들때부터 넣지 않아야함
# 

###########################################################################
print(train_csv.columns)
print(train_csv.info())
print(train_csv.describe())

#모델구성
print(type(train_csv)) #<class 'pandas.core.frame.DataFrame'>

######### train_csv(데이터에서 x와 y를 분리) ############
print("Type:",type[train_csv])

print(train_csv.isnull().sum())
train_csv= train_csv.dropna() #지워버리고 저장을 해아함
print(train_csv.isnull().sum())
print(train_csv.info())
print(train_csv.shape)  #(1328,10)


x= train_csv.drop(['count'],axis=1)  #2개이상은 list
print(x)

y=train_csv['count']
print(y)

x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True , train_size=0.7, random_state=777
)

print(x_train.shape,x_test.shape) #(1021,9)(438,9)->(929,9)(399,9) 
print(y_train.shape,y_test.shape) 

model=Sequential()
model.add(Dense(1, input_dim=9))

model.compile(loss="mse",optimizer="adam")
model.fit(x_train,y_train,epochs=50, batch_size=4)

loss=model.evaluate(x_test,y_test)
print("loss:",loss)