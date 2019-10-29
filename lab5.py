
# coding: utf-8

# In[82]:


from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import recall_score


# In[38]:


digits = load_digits()


# In[39]:


plt.imshow(np.reshape(digits.data[0], (8, 8)), cmap='gray')
plt.title('Label: %i\n' % digits.target[0], fontsize=25)


# In[40]:


x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=1337)


# In[41]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
y_predict = knn.predict(x_test)


# In[42]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[56]:


results = confusion_matrix(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)


# In[58]:



print("Test size = 0.25")
print("KNN Accuracy:",acc)
print(results)


# In[45]:


dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(x_train, y_train)
y_predict_d = dtc.predict(x_test)


# In[70]:


results_d = confusion_matrix(y_test, y_predict_d)
acc_d = accuracy_score(y_test, y_predict_d)


# In[71]:


print(results_d)


# In[72]:


clf_sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=500)


# In[78]:


clf_sgd.fit(x_train,y_train)


# In[79]:


y_predict_sgd = clf_sgd.predict(x_test)
results_sgd = confusion_matrix(y_test, y_predict_sgd)
acc_sgd = accuracy_score(y_test, y_predict_sgd)


# In[80]:


print(results_sgd)


# In[83]:


recall = recall_score(y_test, y_predict, average='macro')
recall_sgd = recall_score(y_test, y_predict_sgd, average='macro')
recall_d = recall_score(y_test, y_predict_d, average='macro')


# In[94]:


print("COMP9517 Week 5 Lab - z5102903\n")
print("Test size = 0.25")
print("random_state = 1337\n")
print("KNN Accuracy:",acc,"  Recall:",recall)
print("SGD Accuracy:",acc_sgd,"  Recall:",recall_sgd)
print("DT Accuracy: ",acc_d,"  Recall:",recall_d)
print("\nBest Performance")
print("KNN Confusion Matrix")
print(results)

