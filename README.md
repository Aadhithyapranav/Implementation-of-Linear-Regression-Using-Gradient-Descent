# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
'''
1.open jupyter notebook
2.open python 3 kernel and write the code in jupyter notebook
3.get the output and post it in the git hub
4.commit changes
'''

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: AADHITHYAA L
RegisterNumber: 212224220003
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Linear Regression using Gradient Descent
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    # Add a column of ones to X for the intercept term
    X = np.c_[np.ones(len(X1)), X1]

    # Initialize theta with zeros
    theta = np.zeros((X.shape[1], 1)).reshape(-1, 1)

    # Perform gradient descent
    for _ in range(num_iters):
        # Calculate predictions
        predictions = X.dot(theta).reshape(-1, 1)

        # Calculate errors
        errors = (predictions - y).reshape(-1, 1)

        # Update theta using gradient descent
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)

    return theta


data = pd.read_csv(r"C:\Users\admin\Downloads\50_Startups.csv", header=None)
print (data.head())
# Assuming the last column is your target variable 'y' and the preceding columns are your features
X = (data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform (X1)
Y1_Scaled = scaler.fit_transform(y)
print('Name:AADHITHYAA L')
print('Register No.:212224220003')
print(X1_Scaled)
print(Y1_Scaled)
# Learn model parameters
theta = linear_regression (X1_Scaled, Y1_Scaled)
# Predict target value for a new data point
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction =np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}") 
*/
```

## Output:
![linear regression using gradient descent](sam.png)

<img width="984" height="400" alt="Screenshot 2025-09-01 173047" src="https://github.com/user-attachments/assets/0883e29a-b51a-48b7-8d9f-ad1264c751e5" />
<img width="703" height="410" alt="Screenshot 2025-09-01 173102" src="https://github.com/user-attachments/assets/09cc998b-55b9-40bf-b75c-f6b9e47fc481" />
<img width="787" height="377" alt="Screenshot 2025-09-01 173113" src="https://github.com/user-attachments/assets/8787570d-5e6f-4a1c-9b89-fdf938bd4c61" />
<img width="565" height="314" alt="Screenshot 2025-09-01 174021" src="https://github.com/user-attachments/assets/890f82db-e35a-4edd-adff-3f86eef65b5b" />







## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
