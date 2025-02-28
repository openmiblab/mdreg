.. _installation:

****************
Installing mdreg
****************

`mdreg` may often be used inside other packages or more complex pipelines, 
and therefore the default installation is lightweight containing only 
essential dependencies. 

This core version of ``mdreg`` can be installed using pip:

.. code-block:: console

   pip install mdreg

This does not include the plot module and only offers `skimage` as a package 
for coregistration. To install the complete version with all possible 
options included, use:

.. code-block:: console

   pip install mdreg[complete]

This installs plotting libraries as well as `antspyx` and `itk-elastix` for 
coregistration. Note ``itk-elastix`` is a heavy package (nearly 800MB).

More fine-grained options are available too:

.. code-block:: console

   pip install mdreg[plot]
   pip install mdreg[elastix]
   pip install mdreg[ants]

And combinations of the above such as:

.. code-block:: console

   pip install mdreg[elastix, ants]

to install all coregistration options but not the plot module.