import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor,plot_tree
import matplotlib.pyplot as plt
import pickle

df=pd.read_csv('Haryana_data.csv')
df.columns = df.columns.str.replace(' ', '')
df['Yield']=df['Production']/df['Area']
df.drop(columns=['Production'],inplace=True)
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
dt=DecisionTreeRegressor(max_depth=30)
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)

r2=r2_score(y_test,y_pred)
print(r2)

if r2>0.8:
    with open('model_pickle','wb') as f:
        pickle.dump(dt,f)


with open('model_pickle','rb') as f:
    dt=pickle.load(f)


state='Haryana'
district='AMBALA'
season='Kharif'
crop='Wheat'
date=2024
area=1254.00



columns=X.columns
df1=pd.DataFrame([X.iloc[0,:]],columns=columns)
df1['Year']=2015
df1[state]=1
df1[crop]=1
df1['Area']=area
df1[district]=1
df1[season]=1
print(df1)


predicted_yield=[]
date=[]
for i in range(2015,2026):
    columns = X.columns
    df1 = pd.DataFrame([X.iloc[0, :]], columns=columns)
    df1['Year'] = i
    df1[state] = 1
    df1[crop] = 1
    df1['Area'] = area
    df1[district] = 1
    df1[season] = 1
    predicted_yield.append(dt.predict(df1))
    date.append(i)



plt.figure(figsize=(10,10))

plt.scatter(date,predicted_yield)
plt.xlabel("The predicted yield")
plt.ylabel("The date")
plt.show()

#plt.figure(figsize=(10,10))
#plot_tree(dt,filled=True,feature_names=list(columns))
#plt.show()

y_new=dt.predict(df1)
print(y_new[0])



