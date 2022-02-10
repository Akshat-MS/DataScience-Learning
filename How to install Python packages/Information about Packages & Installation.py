#!/usr/bin/env python
# coding: utf-8

# # Python Package Installation from Jupyter Notebooks
# 
# - This guide shows you how to install a Python package so that it will work with Jupyter notebook, using pip and/or conda.
# 
# - It will also provide some background as to what exactly the Jupyter notebook abstraction does, how it interacts with the complexities of the operating system, and how you can think about where the "leaks" are, so that you know why things stop working.

# ### pip vs conda
# 
# - pip installs python packages in any environment.
# - conda installs any package in conda environments.

# In[ ]:





# In[ ]:





# In[ ]:





# ### Check the Anaconda version

# In[1]:


conda --version


# ### Pandas package installation

# ##### Installation using pip command

# In[ ]:


get_ipython().system('pip install pandas')


# ##### Upgrade Pandas libraries using PIP command

# In[ ]:


pip install ipython --upgrade


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Pandas Profiling Report

# ##### You can install using the pip package manager by running

# In[ ]:


get_ipython().system('pip install -U pandas-profiling')
jupyter nbextension enable --py widgetsnbextension


# ##### If you are in a notebook (locally, at LambdaLabs, on Google Colab or Kaggle), you can run:

# In[ ]:


import sys
get_ipython().system('{sys.executable} -m pip install -U pandas-profiling[notebook]')
get_ipython().system('jupyter nbextension enable --py widgetsnbextension')


# ### How to install Pandas Profiling package <a href="https://pandas-profiling.github.io/pandas-profiling/docs/master/rtd/pages/installation.html" >Installation steps</a> 

# ## Conda
# 
# ### Myth 1:
# **Conda is a distribution, not a package manager**
# 
# **Reality:** Conda is a package manager; Anaconda is a distribution. Although Conda is packaged with Anaconda, the two are distinct entities with distinct goals.
# 
# A software distribution is a pre-built and pre-configured collection of packages that can be installed and used on a system. A package manager is a tool that automates the process of installing, updating, and removing packages. Conda, with its "conda install", "conda update", and "conda remove" sub-commands, falls squarely under the second definition: it is a package manager
# 
# 
# ### Myth 2:
# **Conda is a Python package manager**
# 
# **Reality:** Conda is a general-purpose package management system, designed to build and manage software of any type from any language. As such, it also works well with Python packages.
# 
# Because conda arose from within the Python (more specifically PyData) community, many mistakenly assume that it is fundamentally a Python package manager. This is not the case: conda is designed to manage packages and dependencies within any software stack. In this sense, it's less like pip, and more like a cross-platform version of apt or yum.
# 
# ### Myth 3: 
# **Conda and pip are direct competitors**
# 
# **Reality:** Conda and pip serve different purposes, and only directly compete in a small subset of tasks: namely installing Python packages in isolated environments.
# 
# Pip, which stands for Pip Installs Packages, is Python's officially-sanctioned package manager, and is most commonly used to install packages published on the Python Package Index (PyPI). Both pip and PyPI are governed and supported by the Python Packaging Authority (PyPA).
# 
# Even setting aside Myth 2, if we focus on just installation of Python packages, conda and pip serve different audiences and different purposes. **If you want to, say, manage Python packages within an existing system Python installation, conda can't help you:** by design, it can only install packages within conda environments. If you want to, say, **work with the many Python packages which rely on external dependencies (NumPy, SciPy, and Matplotlib are common examples), while tracking those dependencies in a meaningful way, pip can't help you:** by design, it manages Python packages and only Python packages.
# 
# ### Myth 4: 
# **Creating conda in the first place was irresponsible & divisive**
# 
# **Reality:** Conda's creators pushed Python's standard packaging to its limits for over a decade, and only created a second tool when it was clear it was the only reasonable way forward
# 
# ### Myth 5: 
# **conda doesn't work with virtualenv, so it's useless for my workflow**
# 
# **Reality:** You actually can install (some) conda packages within a virtualenv, but better is to use Conda's own environment manager: it is fully-compatible with pip and has several advantages over virtualenv
# 
# virtualenv/venv are utilites that allow users to create isolated Python environments that work with pip. Conda has its own <a href="https://conda.io/projects/conda/en/latest/user-guide/concepts/environments.html">built-in environment manager</a> that works seamlessly with both conda and pip, and in fact has several advantages over virtualenv/venv:
# 
#   - conda environments integrate management of different Python versions, including installation and updating of Python itself. Virtualenvs must be created upon an existing, externally managed Python executable.
#   - conda environments can track non-python dependencies; for example seamlessly managing dependencies and parallel versions of essential tools like LAPACK or OpenSSL
#   - Rather than environments built on symlinks – which break the isolation of the virtualenv and can be flimsy at times for non-Python dependencies – conda-envs are true isolated environments within a single executable path.
#   - While virtualenvs are not compatible with conda packages, conda environments are entirely compatible with pip packages. First conda install pip, and then you can pip install any available package within that environment. You can even explicitly list pip packages in conda environment files, meaning the full software stack is entirely reproducible from a single environment metadata file.
#   
# ### Myth 6: 
# **Now that pip uses wheels, conda is no longer necessary**
# 
# **Reality:** wheels address just one of the many challenges that prompted the development of conda, and wheels have weaknesses that Conda's binaries address.

# In[ ]:





# In[ ]:





# In[ ]:





# ### Sources used are - 
# 
# - <a href="https://jakevdp.github.io/blog/2016/08/25/conda-myths-and-misconceptions/">Conda: Myths and Misconceptions</a>
# - <a href="https://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/">Installing Python Packages from a Jupyter Notebook</a>

# In[ ]:




