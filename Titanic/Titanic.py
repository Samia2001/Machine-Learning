import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train=pd.read_csv('C:/Users/HP/Downloads/train.csv')
test=pd.read_csv('C:/Users/HP/Downloads/tested.csv')

import math
def age_crct(row):
    if(np.isnan(row['Age'])):
        if(row['Pclass']==1):
            row['Age']=int(m1)
        elif(row['Pclass']==2):
            row['Age']=int(m2)
        else:
            row['Age']=int(m3)
    return row
def clean(df):
    temp1 = df[df['Pclass']==1]
    temp2 = df[df['Pclass']==2]
    temp3 = df[df['Pclass']==3]
    m1 = math.floor(np.mean(temp1.Age))
    m2 = math.floor(np.mean(temp2.Age))
    m3 = math.floor(np.mean(temp3.Age))
    df = df.apply(age_crct,axis=1)
    df['Age'] = df['Age'].apply(lambda x: math.floor(x))
    df['Fare']=df['Fare'].fillna(df['Fare'].median())
    df=df.drop(['Cabin','Name','Ticket'],axis=1)
    df['Embarked'] = df['Embarked'].fillna('S')
    sex = pd.get_dummies(df['Sex'],drop_first = True)
    embrk=pd.get_dummies(df['Embarked'],drop_first=True)
    df1 = pd.concat([df,sex,embrk],axis=1)
    df1=df1.drop(['Embarked','Sex'],axis=1)
    return df1

train = clean(train)
test = clean(test)

y = train['Survived']
x = train.drop('Survived',axis=1)
y1 = test['Survived']
x1 = test.drop('Survived',axis=1)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x,y)
model.score(x1,y1)


predic = model.predict(x1)
from sklearn.metrics import accuracy_score
accuracy_score(y1,predic)


sub = pd.DataFrame()
sub['PassengerId']=x1['PassengerId']
sub['Survived'] = y1
sub.to_csv('gender_submission.csv',index=False)
