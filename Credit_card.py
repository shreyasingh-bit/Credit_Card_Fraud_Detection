

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("C:/Sample Data/creditcard.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum().sum()


# In[6]:


df.duplicated().sum()


# In[7]:


df.columns 


# In[8]:


plt.plot(df['Amount'],color='y',marker='*')
plt.show
plt.title("Transaction Amount Plot")
plt.xlabel("Transaction Index")
plt.ylabel("Amount")
plt.show()


# In[9]:


plt.scatter(df['Amount'],df['Class'],color='r',marker='*')
plt.title("Amount Vs Class")
plt.xlabel("Transaction Amount")
plt.ylabel("Class(0=Normal and 1=Fraud)")
plt.show()


# In[10]:


plt.boxplot(df['Amount'])
plt.title("Before outlier Removal")
plt.ylabel("Amount")
plt.show()


# In[11]:


df.shape


# In[12]:


#using IQR Method To Remove Outliers
Q1=df['Amount'].quantile(0.25)
Q3=df['Amount'].quantile(0.75)
IQR=Q3-Q1
print("Q1",Q1)
print("Q3",Q3)
print("IQR",IQR)


# In[13]:


lower_limit=Q1-1.5*IQR
upper_limit=Q3+1.5*IQR
print("Lower Limit",lower_limit)
print("Upper Limit",upper_limit)


# In[14]:


df=df[(df['Amount']>=lower_limit)&(df['Amount']<=upper_limit)]
df.shape


# In[15]:


x=df.drop("Class",axis=1)
y=df['Class']


# In[16]:


plt.boxplot(df['Amount'])
plt.title("After outlier Removal")
plt.show()


# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=50)


# In[18]:


print("shape of x_train:",x_train.shape)
print("shape of x_test:",x_test.shape)


# In[19]:


print("shape of y_train:",y_train.shape)
print("shape of y_test:",y_test.shape)


# In[20]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[21]:


sc.fit(x_train)
x_train=sc.transform(x_train)
x_test=sc.transform(x_test)


# In[22]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[23]:


lr.fit(x_train,y_train)


# In[24]:


y_pred=lr.predict(x_test)


# In[25]:


lr.score(x_test,y_test)


# In[26]:


y_pred


# In[27]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


# In[28]:


lr_acc=accuracy_score(y_test,y_pred)
print("Accuracy",lr_acc)


# In[29]:


lr_pr = precision_score(y_test, y_pred)
print("Precision:", lr_pr)


# In[30]:


lr_re=recall_score(y_test,y_pred)
print("Recall",lr_re)


# In[31]:


lr_f1=f1_score(y_test,y_pred)
print("f1_score",lr_f1)


# In[32]:


from sklearn.naive_bayes import GaussianNB
cg=GaussianNB()


# In[33]:


cg.fit(x_train,y_train)


# In[34]:


cg.score(x_test,y_test)


# In[35]:


y_pred=cg.predict(x_test)


# In[36]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


# In[37]:


import seaborn as sns
sns.heatmap(cm,annot=True,fmt='g')


# In[38]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
gnb_acc=accuracy_score(y_test,y_pred)


# In[39]:


print("Accuracy",gnb_acc)


# In[ ]:


gnb_pr = precision_score(y_test, y_pred)
print("Precision:", gnb_pr)


# In[ ]:


gnb_re=recall_score(y_test,y_pred)
print("Recall",gnb_re)


# In[ ]:


gnb_f1=f1_score(y_test,y_pred)
print("f1_score",gnb_f1)


# In[ ]:


metrics=['Accuracy','precision','recall','F1 score']


# In[ ]:


gnb_score=[gnb_acc,gnb_pr,gnb_re,gnb_f1]


# In[ ]:


lr_score=[lr_acc,lr_pr,lr_re,lr_f1]


# In[ ]:


x=np.arange(len(metrics))
width=0.4


# In[ ]:


plt.figure(figsize=(8,5))
plt.bar(x,gnb_score,width,label='GaussianNB',color='blue')
plt.bar(x+width/2,lr_score,width,label='Logistic Regression',color='green')
plt.xlabel("Evalution metrics")
plt.ylabel("score")
plt.title("model comparsion")
plt.xticks(x+width/2,metrics)
plt.legend()
plt.show()


# In[ ]:

import pickle

pickle.dump(lr, open("model.pkl", "wb"))
pickle.dump(sc, open("scaler.pkl", "wb"))
pickle.dump(metrics, open("metrics.pkl", "wb"))
df.sample(500).to_csv("sample.csv", index=False)


# %%
