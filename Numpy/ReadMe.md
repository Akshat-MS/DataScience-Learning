# Basic of NumPy

Data science projects frequently use basic Numpy functions in the projects, which can be found in this file.

Two files have been added, **Numpy Basics.ipynb & Numpy Basics.py.** They both **contain the same information.**. This can be use as a quick reference.


## Numpy
NumPy is a library for scientific computations in Python.Numpy is one of the packages you have to know if you're going to do data science with Python.

It is a Python library that provides **support for large, multidimensional arrays along with masked arrays and matrices**, and **provides extensive functions for performing array manipulation, including mathematical, logical, and shape calculations, sorting, selecting, I/O, discrete Fourier transforms,linear algebra, basic statistical operations, random simulations,** and so on.

**Note** - Explanations are limited to the most frequently used parameters in the function.

Based on their usage, I have categorize the methods.

 - Creating the Numpy Arrays
   - **numpy.array()** - numpy.array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None)
     
     Parameters : 
     - object : array_like : An array, any object exposing the array interface, an object whose __array__ method returns an array,or any (nested) sequence. If object is a scalar, a 0-dimensional array containing object is returned.
       **Example** - list object [1,2,3] or list of tuple [(1,2,3),(4,5,6)]
     - dtype : data-type (optional) - The desired data-type for the array. If not given, then the type will be determined as the minimumtype required to hold the objects in the sequence.
       **Example** - dtype - float or str
 - Inspecting the Numpy Arrays
 - Initializing the Numpy Arrays
 - Datatypes
