import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
df = pd.read_csv('RSMtest_u.csv')
x1 = df['bd']
x2 = df['wl']
y1 = df['umax']
X = []
y = []
for i in range(len(x1)-1):
    X.append([x1[i],x2[i]])
    y.append(y1[i])

X = np.array(X)
y = np.array(y)

model = LinearRegression()
model.fit(X, y)

coefficients = model.coef_
intercept = model.intercept_

e = []
for n in range(len(x1)-1):
    ee = (y1[n]-x1[n]*coefficients[0]+x2[n]*coefficients[1]+intercept)/y1[n]
    e.append(ee)

y_pred = model.predict(X)

r_squared = r2_score(y, y_pred)

print(r_squared)

sns.displot(e, kind='kde')
plt.show()
## 0-520 len(x1)-1

##print(f"回歸方程式：y = {coefficients[0]}x1 + {coefficients[1]}x2 + {intercept}")