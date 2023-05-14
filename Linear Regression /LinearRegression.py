# %% [markdown]
# # Linear Regression -hw1-part2
# #Mohamed Eraky

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_csv('salary_data.csv')
x = df['YearsExperience'].values;
y = df['Salary'].values;
xframe=pd.DataFrame(data=x);
yframe=pd.DataFrame(data=y)

#Plot
df.plot.scatter(x="YearsExperience", y="Salary",c="red")

#generate data frame of ones 
onesframe = pd.DataFrame(np.ones((xframe.size,1)))
A= pd.concat([onesframe, xframe], axis=1)

#computations of LHS
transA=A.T; 
LHS = transA.dot(A);
#compute inverse
LHS = pd.DataFrame(np.linalg.pinv(LHS.values), LHS.columns, LHS.index);
LHS = LHS.dot(transA);
theta_df=LHS.dot(yframe);

# print "thetas"
print("Theta data frame = : ")
print(theta_df)

# convert data frame to numpy
theta=theta_df.to_numpy()

print("Theta = : ")
print(theta)

xx = np.linspace(0,11,100)
yy = theta[1]*xx+theta[0]
plt.plot(xx, yy, '-r')
plt.title('Linear regression')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.grid()
plt.show()


# %%



# %%




