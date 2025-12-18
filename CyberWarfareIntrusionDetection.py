# !/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[3]:


df = pd.read_csv("CyberWarfareIntrusionDetection.csv")
df.head()


# In[4]:


if 'class' in df.columns:
    df.rename(columns={'class': 'label'}, inplace=True)

df.head()


# In[67]:


df.info()
df.describe()
df.shape


# In[68]:


df.isnull().sum()


# In[69]:


sns.countplot(x='label', data=df)
plt.title("Normal vs Malicious Traffic Distribution")
plt.show()


# In[70]:


sns.countplot(x='protocol_type', hue='label', data=df)
plt.title("Attack Distribution by Protocol")
plt.show()


# In[71]:


top_services = df['service'].value_counts().nlargest(10).index

sns.countplot(
    data=df[df['service'].isin(top_services)],
    x='service',
    hue='label'
)
plt.xticks(rotation=45)
plt.title("Top 10 Services: Normal vs Attack")
plt.show()


# In[72]:


sns.countplot(x='flag', hue='label', data=df)
plt.title("Connection Flag vs Traffic Type")
plt.show()


# In[73]:


sns.boxplot(data=df[['duration','src_bytes','dst_bytes','count','srv_count']])
plt.title("Outlier Analysis")
plt.show()


# In[74]:


sns.scatterplot(data=df, x='count', y='srv_count', hue='label', alpha=0.5)
plt.title("Count vs Service Count")
plt.show()


# In[75]:


FEATURES = [
    'protocol_type',
    'service',
    'flag',
    'duration',
    'src_bytes',
    'dst_bytes',
    'count',
    'srv_count'
]

X = df[FEATURES].copy()
y = df['label']


# In[76]:


encoders = {}

for col in ['protocol_type', 'service', 'flag']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le


# In[77]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[78]:


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


# In[79]:


svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)


# In[80]:


y_pred = svm_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


# In[100]:


def predict_intrusion(protocol, service, flag, duration, src_bytes, dst_bytes, count, srv_count):

    input_data = pd.DataFrame([[protocol, service, flag,
                                duration, src_bytes, dst_bytes,
                                count, srv_count]],
                              columns=FEATURES)

    for col in ['protocol_type', 'service', 'flag']:
        input_data[col] = encoders[col].transform(input_data[col])

    input_scaled = scaler.transform(input_data)

    normal_prob, attack_prob = svm_model.predict_proba(input_scaled)[0]

    if attack_prob >= 0.8:
        status = "🚨 MALICIOUS"
    elif attack_prob >= 0.6:
        status = "⚠️ SUSPICIOUS"
    else:
        status = "✅ NORMAL"

    return {
        "status": status,
        "attack_probability_%": round(attack_prob * 100, 2),
        "normal_probability_%": round(normal_prob * 100, 2)
    }


# In[101]:


predict_intrusion(
    protocol='tcp',
    service='http',
    flag='SF',
    duration=60,
    src_bytes=2000,
    dst_bytes=300,
    count=20,
    srv_count=20
)


# In[114]:


predict_intrusion(
    protocol='tcp',
    service='http',
    flag='SF',
    duration=9000,
    src_bytes=2800,
    dst_bytes=8000,
    count=2000,
    srv_count=20
)


# In[103]:


predict_intrusion(
    protocol='tcp',
    service='http',
    flag='SF',
    duration=19000,
    src_bytes=4000,
    dst_bytes=800,
    count=8,
    srv_count=8
)


# In[ ]:




