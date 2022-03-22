#!/usr/bin/env python
# coding: utf-8

# In[1]:


#getting the data
import sagemaker
from sagemaker.local import LocalSession

sagemaker_session = sagemaker.Session()

bucket = sagemaker_session.default_bucket()
prefix = "sagemaker/Demo-pytorch-mnist"

role = sagemaker.get_execution_role()


# In[2]:


role


# In[3]:


pip install torchvision==0.5.0 --no-cache-dir


# In[4]:


from torchvision import datasets, transforms

datasets.MNIST(
    "data",
    download = True,
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)


# In[5]:


inputs = sagemaker_session.upload_data(path='data', bucket=bucket, key_prefix=prefix)
print("input spec (in this case just an S3 path): {}".format(inputs))


# In[8]:


from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point = "mnist.py",
    role = role,
    framework_version = "1.4.0",
    py_version = "py3",
    instance_count = 2,
    instance_type = "ml.c4.xlarge",
    hyoerparameters = {"epochs": 6 , "backend": "gloo"}
)


# In[9]:


estimator.fit({"training": inputs})


# In[10]:


estimator.model_data


# In[11]:


predictor = estimator.deploy(initial_instance_count = 1, instance_type = "ml.t2.medium")


# In[12]:


get_ipython().system('ls data/MNIST/raw')


# In[13]:


import gzip
import numpy as np
import random
import os

data_dir = 'data/MNIST/raw'
with gzip.open(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"), "rb") as f:
    images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32)
    
mask = random.sample(range(len(images)), 16)
mask = np.array(mask, dtype=np.int)
data = images[mask]


# In[17]:


response = predictor.predict(np.expand_dims(data, axis=1))
print('Raw prediction result:')
print(response)
print()

labeled_predictions = list(zip(range(10), response[0]))
print('labeled_predictions:')
print(labeled_predictions)
print()

labeled_predictions.sort(key=lambda label_and_prob: 1.0 - label_and_prob[1])
print("Most lokely answer: {}".format(labeled_predictions[0]))


# In[19]:


from IPython.display import HTML
HTML(open("input.html").read())


# In[23]:


import numpy as np
image = np.array([data], dtype = np.float32)
response = predictor.predict(image)
prediction = response.argmax(axis=1)[0]
print(prediction)


# In[24]:


predictor.delete_endpoint()


# In[ ]:




