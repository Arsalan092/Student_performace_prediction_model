from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

df = pd.read_csv('Student_Grades.csv')
df_copy = df.copy()

# x and y
X = df_copy[['Hours','Practice', 'TeamWork', 'MidTerm', 'FinalExam']]
y = df_copy['Scores']

#train test split 
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size= 0.2, random_state=42)

# linear approach predict
l = LinearRegression()
l_encode = l.fit(X_train,y_train)
pred = l.predict(X_test)

#evaluate 
mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test,pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred) # r2 = 1 - (y_true - pred)^2 / (y_true - y_mean)^2

# figure 

fig, axs = plt.subplots(2,3, figsize=(10,6))

#scatter + regression line 1
axs[0,0].scatter(X_test['Hours'], y_test , color= 'green')
axs[0,0].plot(X_test['Hours'], pred, color = 'red')
axs[0,0].grid(color= 'grey', linestyle=':',linewidth= 0.2)
axs[0,0].set_xlabel('hours')
axs[0,0].set_ylabel('Scores')

#scatter + regression line 2
axs[0,1].scatter(X_test['Practice'], y_test , color= 'blue')
axs[0,1].plot(X_test['Practice'], pred, color = 'red')
axs[0,1].grid(color= 'grey', linestyle=':',linewidth= 0.2)
axs[0,1].set_xlabel('practice')
axs[0,1].set_ylabel('Scores')

#scatter + regression line 3
axs[0,2].scatter(X_test['TeamWork'], y_test , color= 'purple')
axs[0,2].plot(X_test['TeamWork'], pred, color = 'red')
axs[0,2].grid(color= 'grey', linestyle=':',linewidth= 0.2)
axs[0,2].set_xlabel('TeamWork')
axs[0,2].set_ylabel('Scores')

#scatter + regression line 4
axs[1,0].scatter(X_test['MidTerm'], y_test , color= 'Pink')
axs[1,0].plot(X_test['MidTerm'], pred, color = 'red')
axs[1,0].grid(color= 'grey', linestyle=':',linewidth= 0.2)
axs[1,0].set_xlabel('Midterm')
axs[1,0].set_ylabel('Scores')

#scatter + regression line 5
axs[1,1].scatter(X_test['FinalExam'], y_test , color= 'Yellow')
axs[1,1].plot(X_test['FinalExam'], pred, color = 'red')
axs[1,1].grid(color= 'grey', linestyle=':',linewidth= 0.2)
axs[1,1].set_xlabel('FinalExam')
axs[1,1].set_ylabel('Scores')

# off plot
axs[1,2].axis('off')


fig.suptitle('Comparison of Students accuracy', fontsize= 15)
plt.savefig('Student Performance Prediction',dpi= 300, bbox_inches = 'tight')
plt.show()