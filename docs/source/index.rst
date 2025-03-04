#######################
**mdreg** documentation
#######################

Model-driven image registration for medical imaging


***
Aim
***

Pain-free motion-correction of medical image time series with changing 
image contrast. 


********
Features
********

- A simple high-level interface :func:`mdreg.fit` for 
  model-driven registration.

- A collection of :ref:`example data and scripts <examples>` to help new 
  users get started.

- A harmonized interface for  :ref:`coregistration methods <registration>` in 
  three common packages.

- A growing library of :ref:`signal models <models>` for different 
  applications.

- Built-in :ref:`plotting and visualisation options <plot>` for 
  inspection of results.

- Built-in support for larger-than-RAM datasets using zarrays.

- Built-in support for parallel computation using dask.


************
Installation
************

To include the full functionality install with:

.. code-block:: console

    pip install mdreg[complete]

This installs packages needed for plotting and three coregistration 
packages (skimage, antspyx, itk-elastix). 

Alternatively, a lightweight version with core functionality can be installed. 
It does not include plotting options and includes only one package (skimage) 
for coregistration:

.. code-block:: console

    pip install mdreg



*************
Typical usage
*************

Consider a dataset consisting of:

- a 4D array *signal* with a series of free-breathing 3D MRI images of 
  the abdomen with variable flip angles (VFA).
- a 1D array *FA* with the respective flip angles.

To remove the motion with mdreg, we first need to specify which model we 
want to use:

.. code-block:: python

    vfa = {
        'func': mdreg.fit_vfa_lin, 
        'FA': FA,
    }

And that's it. We can now perform motion correction:

.. code-block:: python
  
    coreg, fit, defo, pars = mdreg.fit(signal, fit_image=vfa) 

The function :func:`mdreg.fit` returns 4 arrays:

- *coreg* is the signal array with motion removed;
- *fit* is the array with model fits;
- *defo* is the deformation field;
- *pars* is an array with fitted parameters.

We can inspect the result visually using the built-in plot functions:

.. code-block:: python

    mdreg.plot.animation(coreg)



***************
Getting started
***************

Have look at the :ref:`user guide <user-guide>` or dive straight in with the 
:ref:`examples <examples>`.


******
Citing
******

When you use ``mdreg``, please cite: 

Kanishka Sharma, Fotios Tagkalakis, Irvin Teh, Bashair A Alhummiany, 
David Shelley, Margaret Saysell, Julie Bailey, Kelly Wroe, Cherry Coupland, 
Michael Mansfield, Steven P Sourbron. An open-source, platform independent 
library for model-driven registration in quantitative renal MRI. ISMRM 
workshop on renal MRI, Lisbon/Philadephia, sept 2021.


*******
License
*******

``mdreg`` is distributed under the 
`Apache 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_ license - a 
permissive, free license that allows users to use, modify, and 
distribute the software without restrictions.


.. toctree::
   :maxdepth: 2
   :hidden:
   
   guide/index
   reference/index
   generated/examples/index
   contribute/index
   releases/index
   about/index

