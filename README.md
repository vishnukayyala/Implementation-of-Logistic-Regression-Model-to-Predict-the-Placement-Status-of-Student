# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.

2. Print the placement data and salary data.

3. Find the null and duplicate values.

4. Using logistic regression find the predicted values of accuracy , confusion matrices.

5.  Display the results.

## Program:
```Python
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Vishnu K M
RegisterNumber:  212223240185
*/
import pandas as pd
data1=pd.read_csv('Placement_Data.csv')
data1.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression 
lr=LogisticRegression(solver ="liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

### op1:
![op1](https://github.com/user-attachments/assets/d07cee59-e41b-40af-a0e4-34b2e127d155)

### op2:
![op2](https://github.com/user-attachments/assets/8daa5378-fdd8-41d5-a2fc-a2d2add7b69f)

### op3:
![op3](https://github.com/user-attachments/assets/6099b552-b79c-4789-8097-8e9e9b8d9782)

### op4:
![op4](https://github.com/user-attachments/assets/95a6f42d-cbb5-4140-b5b8-26249dcb9f52)

### op5:
![op5](https://github.com/user-attachments/assets/5dc1e92c-206d-4999-a78c-5cc492c5bd6e)

### op6:
![op6](https://github.com/user-attachments/assets/252280ad-1d4c-4b25-aaf0-bb85b0ae8f7b)

### op7:
![op7](https://github.com/user-attachments/assets/a4dac4b5-1234-4187-b3c7-129d2ae81f9f)

### op8:
![op8](https://github.com/user-attachments/assets/09c13a2f-14f0-4f16-b546-3b971987b57e)

### op9:
![op10](https://github.com/user-attachments/assets/b9cfc82b-187a-4742-ba56-1946b1531dbf)

### op 10:
![op 11](https://github.com/user-attachments/assets/b40f1ff9-74cc-4396-9b0a-917b00171f72)

### op 11:
![op12](https://github.com/user-attachments/assets/82f5add4-33ae-49c4-b495-2653bc1fa14e)

### op 12:
![op13](https://github.com/user-attachments/assets/4a578676-52fc-420c-9152-f7289d334cbc)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
