.. _fit_dict:

*****************************
Signal model fitting
*****************************

Since `mdreg` implements model-driven registration, it is critical to choose 
an appropriate model for any given data type. `mdreg` has a library of 
built-in models that can be be used, and that will grow over time, but also 
offers the option to use a custom-built model. 

The choice of model fit and the parameters to control the optimization can 
be set by providing the *fit_image* dictionary to the `mdreg.fit` function. 
The *fit_image* dictionary has one required item *func* which specifies the 
function to use for model fitting. The other dictionary entries are 
keyword arguments specific to the function.

The *func* entry to *fit_image* be any function, but it must take the image 
array as argument, and return the fit to the model and the fitted model 
parameters. It can be custom built by the user of `mdreg`, or it can be one 
of the built-in fit functions listed under :ref:`fit-funcs`. The functions 
in the list can also be used as a template when building custom functions.

In most cases the signal fit will be performed pixel-by-pixel, meaning the 
signal for each pixel is fitted independently. For such models a more 
convenient alternative is to provide the *fit_pixels* dictionary in the call 
to `mdreg.fit` instead of *fit_image*. 

*fit_pixels* has two required items, one is *model* which defines the 
single-pixel forward model to fit; the other is *p0*, a set of initial values 
of the free model parameters. Beyond that any keyword accepted by 
`mdreg.fit_pixels` can be provided to control the optimization. The function 
will effectively call `scipy.optimize.curve_fit` for pixel-based fitting, 
but includes features such as automatic parallellization for pixel-based 
fitting, sopport for zarrays etc. 

A collection of default forward models is available in mdreg 
(:ref:`signal-models`) and can also be used as templates for building custom 
models.

