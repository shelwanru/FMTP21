#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2
#import os


# In[3]:


def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
        for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
        for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy


# In[4]:


filename = 'gambar_noisy.jpg'

image = cv2.imread('semangka.jpg')
image_noise = noisy('gauss', image)
cv2.imwrite(filename, image_noise) 


# In[5]:


def manhattan(x, y):
    return np.sum(np.absolute(x-y))


# In[6]:


image = cv2.imread('rambutan.jpg')
image2 = cv2.imread('semangka.jpg')

image_noise = cv2.imread('gambar_noisy.jpg')


# In[19]:


dim = (20,20)

image = cv2.resize(image, dim, interpolation = cv2.INTER_LINEAR)
image2 = cv2.resize(image2, dim, interpolation = cv2.INTER_LINEAR)

image_noise = cv2.resize(image_noise, dim, interpolation = cv2.INTER_LINEAR)


# In[9]:


x1 = np.reshape(image, [400,1,3])
x2 = np.reshape(image2, [400,1,3])

y = np.reshape(image_noise, [400,1,3])


# In[10]:


d1 = manhattan(x1, y)
d2 = manhattan(x1, y)


# In[22]:


minimum = np.minimum(d1,d2)

kelas = ''
if d1 == minimum:
    kelas = 'rambutan'
elif d2 == minimum:
    kelas = 'semangka'

print(minimum)
print('Klasifikasi gambar', kelas)


# In[ ]:


cv2.imshow("Foto", image)
cv2.imshow("Foto 2", image2)

cv2.imshow("Foto derau", image_noise)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




