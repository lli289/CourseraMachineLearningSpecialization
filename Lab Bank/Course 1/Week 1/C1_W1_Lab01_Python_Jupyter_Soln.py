#!/usr/bin/env python
# coding: utf-8

# # Optional Lab:  Brief Introduction to Python and Jupyter Notebooks
# Welcome to the first optional lab! 
# Optional labs are available to:
# - provide information - like this notebook
# - reinforce lecture material with hands-on examples
# - provide working examples of routines used in the graded labs

# ## Goals
# In this lab, you will:
# - Get a brief introduction to Jupyter notebooks
# - Take a tour of Jupyter notebooks
# - Learn the difference between markdown cells and code cells
# - Practice some basic python
# 

# The easiest way to become familiar with Jupyter notebooks is to take the tour available above in the Help menu:

# <figure>
#     <center> <img src="./images/C1W1L1_Tour.PNG"  alt='missing' width="400"  ><center/>
# <figure/>

# Jupyter notebooks have two types of cells that are used in this course. Cells such as this which contain documentation called `Markdown Cells`. The name is derived from the simple formatting language used in the cells. You will not be required to produce markdown cells. Its useful to understand the `cell pulldown` shown in graphic below. Occasionally, a cell will end up in the wrong mode and you may need to restore it to the right state:

# <figure>
#    <img src="./images/C1W1L1_Markdown.PNG"  alt='missing' width="400"  >
# <figure/>

# The other type of cell is the `code cell` where you will write your code:

# In[4]:


#This is  a 'Code' Cell
print("This is  code cell")


# ## Python
# You can write your code in the code cells. 
# To run the code, select the cell and either
# - hold the shift-key down and hit 'enter' or 'return'
# - click the 'run' arrow above
# <figure>
#     <img src="./images/C1W1L1_Run.PNG"  width="400"  >
# <figure/>
# 
#  

# ### Print statement
# Print statements will generally use the python f-string style.  
# Try creating your own print in the following cell.  
# Try both methods of running the cell.

# In[5]:


# print statements
variable = "right in the strings!"
print(f"f strings allow you to embed variables {variable}")


# # Congratulations!
# You now know how to find your way around a Jupyter Notebook.

# In[ ]:




