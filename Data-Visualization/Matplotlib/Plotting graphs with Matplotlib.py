#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

# Importing matplotlib and seaborn for graphs
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Line Plot

# In[3]:


plt.plot([3,5,6])


# In[4]:


plt.plot([3,5,6])
plt.show()


# In[5]:


plt.plot([10,11,12],[3,5,6])
plt.show()


# In[6]:


x = np.arange(1,10)


# In[7]:


y = x ** 2


# In[8]:


x


# In[9]:


y


# In[10]:


plt.plot(x,y)
plt.show()


# In[11]:


plt.plot(x,y,"r")
plt.title("Square")
plt.xlabel("x-values")
plt.ylabel("y-values")
plt.show()


# In[12]:


plt.plot(x,y,"r*")
plt.title("Square")
plt.xlabel("x-values")
plt.ylabel("y-values")
plt.show()


# ### Change the line pattern and Color

# In[13]:


# *,o,v,^,-,--,.
plt.plot(x,y,"r^")
plt.title("Square")
plt.xlabel("x-values")
plt.ylabel("y-values")
plt.show()


# In[14]:


# *,o,v,^,-,--,.
plt.plot(x,y,"r--")
plt.title("Square")
plt.xlabel("x-values")
plt.ylabel("y-values")
plt.xlim(0,10)
plt.ylim(-20,120)
plt.show()


# ### Adding labels, title and limit the axis

# In[15]:


# *,o,v,^,-,--,.
plt.plot(x,3*x+5,label="y==x")
plt.plot(x,y,"r--",label="y=x^2")
plt.title("Square")
plt.xlabel("x-values")
plt.ylabel("y-values")
plt.xlim(0,10)
plt.legend()
plt.show()


# In[16]:


# *,o,v,^,-,--,.
plt.plot(x,3*x+5,label="y==x")
plt.plot(x,y,"r--",label="y=x^2")
plt.title("Square")
plt.xlabel("x-values")
plt.ylabel("y-values")
plt.text(3.1,23,"(4.1,21)")
plt.xlim(0,10)
plt.legend()
plt.show()


# In[17]:


# *,o,v,^,-,--,.
plt.plot(x,3*x+5,label="y==x")
plt.plot(x,y,"r--",label="y=x^2")
plt.title("Different Methods Practise")
plt.xlabel("x-values")
plt.ylabel("y-values")
plt.text(3.1,23,"(4.1,21)")
plt.legend()
plt.grid()
plt.show()


# In[18]:


plt.plot(x,y,"r--",x,3*x+20,"g-",x,3*x*x/2,"yo-")
plt.show()


# In[19]:


p = np.arange(0.0001,100)
plt.plot(p,np.log(p),"g-")
plt.show()


# In[20]:


import math
math.log(10)


# In[21]:


y = [math.log(i) for i in x]
m = list(map(lambda x: math.log(x),x))
n = list(map(math.log,x))


# ## Vectorization

# In[22]:


np_log = np.vectorize(math.log)


# ## Bar plot

# In[23]:


plt.bar([1,2,3],[10,20,5])
plt.xlabel("x-values")
plt.show()


# In[24]:


plt.bar([1,2,3],[10,20,5],width=0.2,color="g",label="News Rating")
plt.xlabel("x-values")
plt.legend()
plt.show()


# In[25]:


plt.bar([1,2,3],[10,20,5],width=0.2,color="b",label="News Rating")
plt.bar([1,2,3],[5,10,15],width=0.2,color="r",label="Sales")
plt.xlabel("x-values")
plt.legend()
plt.show()


# In[26]:


plt.bar([1,2,3],[10,20,5],width=0.2,color="g",label="News Rating")
plt.bar([1.2,2.2,3.2],[5,10,15],width=0.2,color="r",label="Sales") # offset 0.2
plt.xlabel("x-values")
plt.legend()
plt.show()


# In[27]:


plt.bar([1,2,3],[10,20,5],width=0.2,color="g",label="News Rating")
plt.bar(np.array([1,2,3])+0.2,[5,10,15],width=0.2,color="y",label="Sales") # offset 0.2
plt.xlabel("x-values")
plt.legend()
plt.show()


# ## Histoograms

# In[28]:


vals = [1,2,3,2,4,3,5,2,2,1,3,4,5,1.7,1.3,3.1,2.1,3.6]
plt.hist(vals)
plt.show()


# In[29]:


vals = [1,2,3,2,4,3,5,2,2,1,3,4,5,1.7,1.3,3.1,2.1,3.6]
cnt,bins,patches = plt.hist(vals,bins=5)
plt.show()


# In[30]:


print(cnt)
print(bins,len(bins))
print(patches)


# In[31]:


mu = 100
sigma = 15
s = np.random.normal(mu,sigma,100000)
plt.hist(s,bins=100)
plt.show()


# ## Inference
#   - Most of the values are close to 100, mean=1000
#   - 70% of data is present under 1 standard deviation(85-115)
#   - 98% of the data oresent under 2 standard deviations (70-130))
#   - 99.9% of data lies under 3 std , (55,145)

# In[32]:


mu = 100
sigma = 100
s = np.random.normal(mu,sigma,100000)
plt.hist(s,bins=100)
plt.show()


# ## Scatter plot

# In[33]:


x = np.random.rand(20)
y = np.random.rand(20)
plt.scatter(x,y)
plt.show()


# In[34]:


x = np.random.rand(20)
y = np.random.rand(20)
x1 = np.random.rand(20)
y1 = np.random.rand(20)
plt.scatter(x1,y1,color="r",label="x and y")
plt.scatter(x,y,label="x2 and y1")
plt.legend()
plt.show()


# ## Pie Chart

# In[35]:


plt.pie([1,2,3,4],
    labels = ["Law","Education","Health","Defense"],
    colors = ["r","g","b","y"],startangle=90,explode=[0,0.2,0,0]
       , shadow = True)
plt.show() 


# In[ ]:





# ### Sigmod Function

# In[36]:


x = np.arange(-100,100)


# In[37]:


# sigmod = y = [math.log(i) for i in x]
# m = list(map(lambda x: math.log(x),x))
# n = list(map(math.log,x))


# In[38]:


sigmod = 1 / (1 + np.exp(-x))


# In[39]:


plt.plot(x,sigmod,"g-")
plt.title("Sigmod")
plt.show()


# In[ ]:





# In[40]:


def np_sigmod(x):
    return 1/(1+math.exp(-x))


# In[41]:


sigmod = np.vectorize(np_sigmod)
x = np.arange(-100,100)
y = sigmod(x)


# In[42]:


plt.plot(x,y)
plt.show()


# In[43]:


plt.plot(x,y)
plt.xlim(-10,10)
plt.show()


# ### Sub plots

# In[44]:


x = np.arange(1,10,0.1)
y = np.sin(x)
plt.plot(x,y)


# In[45]:


x = np.arange(1,10,0.1)
sin = np.sin(x)
cos = np.cos(-x)

plt.figure(figsize=(5,5))
plt.subplot(2,2,1)
plt.plot(x,sin)

plt.subplot(2,2,4)
plt.plot(x,cos)

plt.show()


# In[46]:


x = np.arange(1,10,0.1)
sin = np.sin(x)
tan = np.tan(x)
log = np.log(x)

plt.figure(figsize=(8,8))
plt.subplot(2,3,1)
plt.plot(x,sin)

plt.subplot(2,3,3)
plt.plot(x,tan)

plt.subplot(2,3,5)
plt.plot(x,log)

plt.show()


# In[47]:


fig = plt.figure()

ax231 = fig.add_subplot(231)
ax231.plot(x,y)
ax233 = fig.add_subplot(233)
ax233.plot(x,tan)
ax234 = fig.add_subplot(234)
ax234.plot(x,log)
plt.show()


# In[48]:


a = np.array([0,1,2,3])
b = np.array([0,1,2])


# In[49]:


ax,by = np.meshgrid(a,b)


# In[50]:


ax


# In[51]:


by


# ## Mesh
# (0,0),(1,0),(2,0),(3,0)
# (0,1), (1,1), (2,1), (3,1)
# (0,2), (1,2),(2,2),(3,2)

# In[52]:


fig = plt.figure()
axis = fig.add_subplot(projection="3d")
axis.plot_surface(ax,by,np.array([[7]]))
plt.show()


# In[53]:


z = ax**2 + by**2


# In[54]:


fig = plt.figure()
axis = fig.add_subplot(projection = '3d')
axis.plot_surface(ax,by,z)
plt.show()


# In[55]:


x4 = np.arange(-1,1,0.0005)
y4 = x4
x4,y4 = np.meshgrid(x4,y4)
z = x4**2 + y4**2

fig = plt.figure()
axis = fig.add_subplot(projection = '3d')
axis.plot_surface(x4,y4,z)
plt.show()


# In[56]:


img = plt.imread('./resized_nearest.jpg')


# In[57]:


type(img)


# In[58]:


img.shape


# In[62]:


plt.imshow(img)
#plt.axis("off")
plt.show()


# In[80]:


# face = img[row,col,:]
face = img[40:210,70:240,:]


# In[81]:


plt.imshow(face)
plt.show()


# In[82]:


plt.imsave("./cat_face.jpeg",face)


# ### Adding border

# In[102]:


img = plt.imread('./resized_nearest.jpg')


# In[103]:


img.shape


# In[116]:


# Function to add border
def add_borders(image, border_width=20):
    img_cpy = image.copy()
    img_cpy = np.hstack((np.zeros((img_cpy.shape[0], border_width, 3), dtype="int8"), img_cpy))
    img_cpy = np.vstack((img_cpy, np.zeros((border_width, img_cpy.shape[1], 3), dtype="int8")))
    img_cpy = np.hstack((img_cpy, np.zeros((img_cpy.shape[0], border_width, 3), dtype="int8")))
    img_cpy = np.vstack((np.zeros((border_width, img_cpy.shape[1], 3), dtype="int8"), img_cpy))
    return img_cpy


# In[117]:


# Add border
border_img = add_borders(img)
plt.imshow(border_img)


# In[118]:


res_img = border_img.reshape(-1, 3)
res_df = pd.DataFrame(columns=["c1", "c2", "c3"])
res_df["c1"] = res_img[:, 0]
res_df["c2"] = res_img[:, 1]
res_df["c3"] = res_img[:, 2]
res_df.to_csv('./image_border.csv',index=False)


# In[119]:


res_df.shape


# In[99]:


# image_b = np.pad(array=img1, pad_width=20, mode='constant', constant_values=0)


# In[120]:


# image_b.shape


# In[113]:


# plt.imshow(image_b)
# # plt.imshow(pm)
# #plt.axis("off")
# plt.show()# 


# In[112]:


# color = [101, 52, 152] # 'cause purple!
# top, bottom, left, right = [150]*4
# img_with_border = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


# In[ ]:




