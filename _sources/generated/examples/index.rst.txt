:orphan:

.. _examples:

########
Examples
########


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>


*********
Tutorials
*********





.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example illustrates how mdreg can be used to fit models without  motion correction. The idea is illustrated for a 3D time series with  variable flip angles (VFA), and a linear signal model fit.">

.. only:: html

  .. image:: /generated/examples/tutorials/images/thumb/sphx_glr_plot_model_fitting_thumb.gif
    :alt:

  :ref:`sphx_glr_generated_examples_tutorials_plot_model_fitting.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Fitting models without motion correction</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="By default, mdreg performs coregistration using the optical flow method  optical_flow_tvl1 as implemented in skimage, with default  settings for all parameters. ">

.. only:: html

  .. image:: /generated/examples/tutorials/images/thumb/sphx_glr_plot_coreg_options_thumb.gif
    :alt:

  :ref:`sphx_glr_generated_examples_tutorials_plot_coreg_options.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Coregistration options in mdreg</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="By default mdreg performs 3D coregistration on 3D series, and 2D  coregistration on 2D series. In case the 3D series encodes a collection of  2D slices with gaps in between (multi-slice data), this may not be appropriate.">

.. only:: html

  .. image:: /generated/examples/tutorials/images/thumb/sphx_glr_plot_multislice_thumb.png
    :alt:

  :ref:`sphx_glr_generated_examples_tutorials_plot_multislice.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multi-slice 2D registration</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Since mdreg performs model-driven motion correction, the choice of an  appropriate model is important to the result. We illustrate this here by coregistering a dataset with different models.">

.. only:: html

  .. image:: /generated/examples/tutorials/images/thumb/sphx_glr_plot_model_selection_thumb.gif
    :alt:

  :ref:`sphx_glr_generated_examples_tutorials_plot_model_selection.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The effect of model selection</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="When data are entered as numpy arrays, mreg will load the entire array in memory,  perform the computations and generate new datasets in memory, such as model fit  arrays or deformation fields. ">

.. only:: html

  .. image:: /generated/examples/tutorials/images/thumb/sphx_glr_plot_large_data_thumb.png
    :alt:

  :ref:`sphx_glr_generated_examples_tutorials_plot_large_data.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Datasets larger than RAM</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In case an appropriate signal model is not available in mdreg&#x27;s model  library, the model must be custom writtem. This tutorial will illustrate this  for the case of T1-mapping with a MOLLI sequence.">

.. only:: html

  .. image:: /generated/examples/tutorials/images/thumb/sphx_glr_plot_models_options_thumb.gif
    :alt:

  :ref:`sphx_glr_generated_examples_tutorials_plot_models_options.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Defining a signal model</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="mdreg includes a harmonized API for pairwise coregistration of 2D images or 3D volumes with three different packages: ants, skimage and  itk-elastix. ">

.. only:: html

  .. image:: /generated/examples/tutorials/images/thumb/sphx_glr_plot_coreg_pairwise_thumb.png
    :alt:

  :ref:`sphx_glr_generated_examples_tutorials_plot_coreg_pairwise.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Pairwise coregistration API</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="mdreg includes a harmonized API for groupwise coregistration of  series of 2D images or 3D volumes with three different packages: ants,  skimage and itk-elastix. This examples illustrates their usage.">

.. only:: html

  .. image:: /generated/examples/tutorials/images/thumb/sphx_glr_plot_coreg_groupwise_thumb.gif
    :alt:

  :ref:`sphx_glr_generated_examples_tutorials_plot_coreg_groupwise.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Groupwise coregistration API</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

*********
Use cases
*********

A collection of real-world applciations of the mdreg package.





.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows how mdreg can be used to perform 2D motion correction  slice-by-slice on a 4D array, for the case of Look-Locker T1 mapping.">

.. only:: html

  .. image:: /generated/examples/use_cases/images/thumb/sphx_glr_plot_molli_kidney_thumb.gif
    :alt:

  :ref:`sphx_glr_generated_examples_use_cases_plot_molli_kidney.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multi-slice MOLLI T1-mapping in the kidney</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Fit a linear 2-compartment model to a 3D DCE dataset. ">

.. only:: html

  .. image:: /generated/examples/use_cases/images/thumb/sphx_glr_plot_dce_liver_thumb.png
    :alt:

  :ref:`sphx_glr_generated_examples_use_cases_plot_dce_liver.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D DCE-MRI in the liver</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example illustrates motion correction of a 3D time series with  variable flip angles (VFA). The motion correction is performed with 3D  coregistration and using a linear signal model fit.">

.. only:: html

  .. image:: /generated/examples/use_cases/images/thumb/sphx_glr_plot_vfa_liver_thumb.gif
    :alt:

  :ref:`sphx_glr_generated_examples_use_cases_plot_vfa_liver.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D Variable Flip Angle (Linear)</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:
   :includehidden:


   /generated/examples/tutorials/index.rst
   /generated/examples/use_cases/index.rst



.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
