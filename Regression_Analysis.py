import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
df = pd.read_csv('RSMtest_u.csv')
x1 = df['bd']
x2 = df['wl']
y1 = df['umax']
X = []
y = []
for i in range(len(x1)-1):
    X.append([x1[i],x2[i]])
    y.append(y1[i])

model = LinearRegression()
model.fit(X, y)

coefficients = model.coef_
intercept = model.intercept_

e = (y1[10]-coefficients[0]*x1[10]-coefficients[1]*x2[10]-intercept)/y1[10]
## 0-520 len(x1)-1
print(e)
print(f"回歸方程式：y = {coefficients[0]}x1 + {coefficients[1]}x2 + {intercept}")