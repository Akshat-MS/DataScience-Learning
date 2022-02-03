#!/usr/bin/env python
# coding: utf-8

# # NumPy
# 
# 

# In[32]:


# NumPy is a library for scientific computations in Python.
# Numpy is one of the packages you have to know if you're going to do data science with Python.

# It is a Python library that provides support for large, multidimensional arrays along with masked 
# arrays and matrices, and provides extensive functions for performing array manipulation, including 
# mathematical, logical, and shape calculations, sorting, selecting, I/O, discrete Fourier transforms,
# linear algebra, basic statistical operations, random simulations, and so on.


# In[3]:


# Importing the Numpy library
import numpy as np


# In[5]:


# Numpy version
np.__version__


# # Function - numpy.array()

# In[6]:


# Function - numpy.array() (Note only most frequently parameters are explain below)
# numpy.array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None)

# Parameters : 
# object : array_like
# An array, any object exposing the array interface, an object whose __array__ method returns an array,
# or any (nested) sequence. If object is a scalar, a 0-dimensional array containing object is returned.
# Example - list object [1,2,3] or list of tuple [(1,2,3),(4,5,6)]

# dtype : data-type (optional)
# The desired data-type for the array. If not given, then the type will be determined as the minimum
# type required to hold the objects in the sequence.
# Example - dtype - float or str


# In[7]:


a = np.array([1,2,3])
b = np.array([(1,2,3),(6,7,8)],dtype = float)
c = np.array([[(1,2,3),(4,5,6)],[(7,8,9),(10,11,12)]], dtype= str)


# In[8]:


print(a, " \t" , type(a))
print("\n",b, " \t" , type(b))
print("\n",c, " \t" , type(c))


# # Numpy supports -  element-wise operations

# In[9]:


# When a*2 is used, it performs element-wise operations rather than duplicating the content as with lists.
print(a*2)
print(b*2)


# # Function required for inspecting the Numpy Array
#   - a.shape
#   - a.ndim
#   - a.size
#   - a.dtype
#   - a.dtype.name
#   - a.astype(float)

# In[10]:


# a.shape - Tuple of array dimensions.
## The shape property is usually used to get the current shape of an array
## Example - (3,) - What is the magnitude of each dimension.
c.shape


# In[11]:


# a.ndim - Dimension of the array (1d,2d or nth array)
## Example - a.ndim - Array of dimension 1 (1d).
c.ndim


# In[12]:


# a.size - Number of elements in the array.
## Equal to np.prod(a.shape), i.e., the product of the array’s dimensions.
## Example c.size - shape of c is (2,2,3) = 2*2*3 = 12 
c.size


# In[13]:


# a.dtype - Data-type of the array’s elements.
## Example a.dtype - dtype('int32') and b.dtype - dtype('float64')
b.dtype


# In[14]:


# a.dtype.name - A bit-width name for this data-type.
## Un-sized flexible data-type objects do not have this attribute.
## Example x = np.dtype(float) x.name is float64
x = np.dtype(float)
y = np.dtype([('a', np.int32, 8), ('b', np.float64, 6)])
print(x.name,y.name)


# In[15]:


# a.astype(float)
## Convert the array's datatype according to the parameter given
## b.astype - When creating the array b, we used float, and then used astype to convert to integer.
b.astype(int)


# # Different methods to Initialize the numpy array

# ## numpy.arange

# In[16]:


# numpy.arange([start, ]stop, [step, ]dtype=None, *, like=None)
## Return evenly spaced values within a given interval.
## Values are generated within the half-open interval [start, stop) 
## (In other words, the interval including start but excluding stop). 
## For integer arguments the function is equivalent to the Python built-in range function, 
## but returns an ndarray rather than a list.
print(np.arange(0,10,2,dtype = int))
print(np.arange(0,10,0.5,dtype=float))


# ## numpy.linspace

# In[17]:


# numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)[source]
## Return evenly spaced numbers over a specified interval.
## Returns num evenly spaced samples, calculated over the interval [start, stop].
## The endpoint of the interval can optionally be excluded.
print(np.linspace(2.0, 3.0, num=5))
print(np.linspace(2.0, 3.0, num=5, endpoint=False))
print(np.linspace(2.0, 3.0, num=5, retstep=True))


# ## Difference between the linspace and arange.

# In[18]:


# Difference between the linspace and arange.
## arange allow you to define the size of the step. linspace allow you to define the number of steps.
## Example where arange might fail are - 
print(" Using arange ",np.arange(0, 5, 0.5, dtype=int))
print(" Using arange ",np.arange(-3, 3, 0.5, dtype=int))

print(" Using Linspace ", np.linspace(0, 5, num = 5))
print(" Using Linspace ",np.linspace(-3, 3,num = 5))


# ## Difference between numpy Array and Lists

# |Numpy Array|List| 
# |-----|-------|
# |Numpy data structures consume less memory|List take more memory than numpy array| 
# |Numpy are faster |Lists are slower as compared to numpy array|
# |NumPy have optimized functions such as linear algebra operations built in||
# |Element wise operation is possible||
# |Array are by default Homogeneous, which means data inside an array must be of the same Datatype.|A list can store different data types|
# ||A list can consist of different nested data size|
# |We can create a N-dimensional array in python using numpy.array().||
# ||A list is easier to modify|
# |Array can handle mathematical operations|A list cannot directly handle a mathematical operations|

# ## numpy.zeros

# In[25]:


# numpy.zeros(shape, dtype=float, order='C', *, like=None)
## Return (ndarray) a new array of given shape and type, filled with zeros.
arr_1d_zeros = np.zeros(5)
arr_2d_zeros = np.zeros((2,5),dtype="int64")
arr_3d_zeros = np.zeros((2,3,4),dtype = int)


# In[31]:


print("\n 1D Array\n",arr_1d_zeros)
print("\n 2D Array\n", arr_2d_zeros)
print("\n 3D Array\n",arr_3d_zeros)


# ## numpy.ones

# In[32]:


# numpy.ones(shape, dtype=None, order='C', *, like=None)
## Return a new array of given shape and type, filled with ones.
arr_1d_ones = np.ones(5)
arr_2d_ones = np.ones((2,5),dtype="int64")
arr_3d_ones = np.ones((2,3,4),dtype = int)


# In[33]:


print("\n 1D Array\n",arr_1d_ones)
print("\n 2D Array\n", arr_2d_ones)
print("\n 3D Array\n",arr_3d_ones)


# ## numpy.full

# In[35]:


# numpy.full(shape, fill_value, dtype=None, order='C', *, like=None)
## Return a new array of given shape and type, filled with fill_value.
arr_1d_full = np.full(2, np.inf)
arr_2d_full = np.full((2, 2), 5)
arr_3d_full = np.full((2, 2,2), [1, 2])


# In[36]:


print("\n 1D Array\n",arr_1d_full)
print("\n 2D Array\n", arr_2d_full)
print("\n 3D Array\n",arr_3d_full)


# ## numpy.eye

# In[37]:


# numpy.eye(N, M=None, k=0, dtype=<class 'float'>, order='C', *, like=None)
## Return a 2-D array with ones on the diagonal and zeros elsewhere
arr_2d_diag0 = np.eye(2, dtype=int)
arr_2d_diag1 = np.eye(3, k=1)


# In[39]:


print("\n 2D Array\n", arr_2d_diag0)
print("\n 2D Array\n", arr_2d_diag1)


# ## random.random

# In[40]:


# random.random(size=None)
## Return random floats in the half-open interval [0.0, 1.0). 
## Alias for random_sample to ease forward-porting to the new random API.
np.random.random()


# # Array Manipulation
# 

# ## numpy.transpose

# In[55]:


# numpy.transpose(a, axes=None)
# Reverse or permute the axes of an array; returns the modified array.
# Parameters | aarray_like | Input array.

# axes | tuple or list of ints, optional
# If specified, it must be a tuple or list which contains a permutation of [0,1,..,N-1] where N is the 
# number of axes of a. The i’th axis of the returned array will correspond to the axis numbered 
# axes[i] of the input. If not specified, defaults to range(a.ndim)[::-1], which reverses the order 
# of the axes.
x = np.arange(4).reshape((2,2))
print(x)
np.transpose(x)


# In[59]:


x = np.ones((1, 2, 3))
print("Original Array\n",x)
print("\n After Transpose \n",np.transpose(x, (1, 0, 2)))


# ## numpy.vstack

# In[44]:


# numpy.vstack(tup)
# Stack arrays in sequence vertically (row wise).
# Parameters | tup | sequence of ndarrays
# The arrays must have the same shape along all but the first axis. 
# 1-D arrays must have the same length.

# Returns | stacked | ndarray
# The array formed by stacking the given arrays, will be at least 2-D.

top_stack = np.linspace(0, 3, 4).reshape(2,2)
bottom_stack = np.linspace(5, 8, 4).reshape(2,2)
vstack = np.vstack((top_stack,bottom_right))


# In[47]:


print("Array \n",top_stack)
print("\nArray \n",bottom_right)
print("\nMerged Array \n",vstack)


# ## numpy.hstack

# In[48]:


# numpy.hstack(tup)
# Stack arrays in sequence horizontally (column wise).
# Parameters | tup | sequence of ndarrays
# The arrays must have the same shape along all but the second axis, 
#except 1-D arrays which can be any length.

# Returns | stacked | ndarray
# The array formed by stacking the given arrays.
left_stack = np.linspace(0, 3, 4).reshape(2,2)
right_stack = np.linspace(5, 8, 4).reshape(2,2)
hstack = np.hstack((left_stack,right_stack))


# In[49]:


print("Array \n",left_stack)
print("\nArray \n",right_stack)
print("\nMerged Array \n",hstack)


# ## numpy.concatenate

# In[51]:


# numpy.concatenate((a1, a2, ...), axis=0, out=None, dtype=None, casting="same_kind")
# Join a sequence of arrays along an existing axis

# Parameters | a1, a2, …sequence of array_like
# The arrays must have the same shape, except in the dimension corresponding to axis 
# (the first, by default).

# axis | int, optional
# The axis along which the arrays will be joined. If axis is None, arrays are flattened before use. 
# Default is 0.

# Returns | res | ndarray | he concatenated array.

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
np.concatenate((a, b), axis=0)


# In[52]:


np.concatenate((a, b.T), axis=1)


# In[53]:


np.concatenate((a, b), axis=None)


# In[ ]:




