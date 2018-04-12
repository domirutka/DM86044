
# coding: utf-8

# In[5]:


from sklearn.datasets import load_iris 


# In[6]:


iris=load_iris()


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


import pandas as pd


# In[10]:


import numpy as np


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)


# In[12]:


X_train.shape


# In[13]:


X_test.shape


# In[14]:


from sklearn.neighbors import KNeighborsClassfier


# In[15]:


from sklearn.neighbors import KNeighborsClassifier


# In[17]:


ds=KNeighborsClassifier(3)


# In[18]:


ds.fit(X_train, y_train)


# In[19]:


y_pred=ds.predict(X_test)


# In[25]:


from sklearn.metrics import accuracy_score, classification_report


# In[26]:


print(accuracy_score(y_pred, y_test))


# In[27]:


print(classification_report(y_pred, y_test))


# In[28]:


import numpy as np


# In[29]:


total_rzuty = 30


# In[30]:


liczba_orlow = 24


# In[41]:


prawd_orla = 0.5


# In[42]:


experiment = np.random.randint(0,2,total_rzuty)


# In[43]:


print("Dane Eksperymentalne :{}".format(experiment))


# In[39]:


ile_orlow = experiment[experiment==1].shape[0]


# In[45]:


print("Liczba orłów w eksperymencie:", ile_orlow )


# In[50]:


def rzut_moneta_eksperyment(ile_razy_powtorzyc):
    head_count = np.empty([ile_razy_powtorzyc,1],dtype=int)
    for times in np.arange(ile_razy_powtorzyc):
        experiment = np.random.randint(0,2, total_rzuty)
        head_count[times] = experiment[experiment==1].shape[0]
    return head_count


# In[70]:


head_count = rzut_moneta_eksperyment(10000)
head_count[:10]
print('Wymiar:{} \n Typ: {}'.format(head_count.shape,type(head_count)))


# In[71]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(color_codes = True)

sns.distplot(head_count, kde=False)
sns.distplot(head_count, kde=True)


# In[72]:


print('Otrzymaliśmy {} 24 i wiecej orłów. Co stanowiło {} procent'.format(head_count[head_count >= 24].shape[0],(head_count[head_count>=24].shape[0]/float(head_count.shape[0])*100)))


# In[73]:




(head_count[head_count>=24].shape[0]/float(head_count.shape[0])*100)


# In[78]:


def coin_toss_experiment(times_to_repeat):

    head_count = np.empty([times_to_repeat,1], dtype=int)
    experiment = np.random.randint(0,2,[times_to_repeat,total_rzuty])
    return experiment.sum(axis=1)


# In[79]:


coin_toss_experiment(10)

