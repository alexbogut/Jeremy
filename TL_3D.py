import numpy as np

from sklearn.svm import SVR
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from adapt.instance_based import TrAdaBoostR2

#change for different layer heigths
filename = r'printer_widths_70_3d.xlsx'
df = pd.read_excel(filename)

# b = list(df['layer_height'])
c = list(df['feed_rate'])
z = list(df['stage_speed'])
yt = list(df['target'])
ys = list(df['source'])
arrx = [c,z]
yt = np.array(yt)
ys = np.array(ys)
X = np.array(arrx)
#X = DataFrame(X)
X = X.transpose()
c = np.array(c)
z = np.array(z)
C = c
Z = z
c = np.sort(c)
z = np.sort(z)

c, z = np.meshgrid(c, z)



filename = r'target_points_70_3d.xlsx'
df = pd.read_excel(filename)

# b = list(df['layer_height'])
g = list(df['feed_rate'])
p = list(df['stage_speed'])
Yt = list(df['target'])
Ys = list(df['source'])
Arrx = [g,p]
Yt = np.array(Yt)
Ys = np.array(Ys)
x = np.array(Arrx)
#X = DataFrame(X)
x = x.transpose()
g = np.array(g)
p = np.array(p)
G = g
P = p
g = np.sort(g)
p = np.sort(p)






#SVR Model Fit on training data for 0.7mm layer height
svr = SVR(kernel="rbf", C=2, epsilon=0.0001)
# svr.fit(x_train,y_train)




#ADABOOOST
tr = TrAdaBoostR2(svr, n_estimators=30, random_state=0)
tr.fit(X, ys.reshape(-1, 1), x, Yt.reshape(-1, 1))


ADAfit = tr.predict(X)



score = mean_absolute_percentage_error(yt, ADAfit)
print("MAPE:", score)
mse = mean_squared_error(yt, ADAfit)
print("MSE:", mse)



#Plotting Code
y_plot_tr = tr.predict(np.c_[c.ravel(), z.ravel()]).reshape(c.shape)


ax = plt.figure().gca(projection='3d')
ax.set_xlim(125,750)
ax.set_ylim(300, 750)
ax.set_xticks(np.arange(125, 750, 100))
ax.set_yticks(np.arange(300, 750, 100))
surf = ax.plot_surface(c, z, y_plot_tr, rstride=3, cstride=3, color='red', alpha=0.4)
points = ax.scatter(C, Z, yt)
ax.set_xlabel('$Feed Rate (mm/min)$', fontsize=10, rotation=150)
ax.set_ylabel('$Stage speed (mm/min)$')
ax.set_zlabel('$Line Width (mm)$', fontsize=10, rotation=60)
ax.yaxis._axinfo['label']['space_factor'] = 3.0
# plt.title('4D Layers MAPE: '+str(score))
plt.show()