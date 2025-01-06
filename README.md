# Jabble
Jabble is a math modeling package designed for fitting stellar radial velocities to spectrograph data.
The models can be divided into two classes in this package. There are ContainerModels that are composed of SubModels that perform an operation with given parameters.

## Installation
To install a dev version, run...
`git clone git+https://github.com/mdd423/wobble_jax.git`
Then run,
`cd wobble_jax`
`pip install .`
Or to install a stable release, run
`pip install jabble`


## Quickplay
For quick use, the quickplay module is made to create the models necessary for spectral template fitting and RV extraction. Use 'WobbleModel' for generating a model with stellar and telluric components. Or use 'PseudoNormalModel' to generate a model with stellar, telluric, and pseudo-normalizing components. 

Then use 'model.save' function to save the object parameters to hdf5 files. The stellar model will then compute the fisher information for the RV parameters. Be wary of bad fits contributing to the RV information, very bad fits can have erroneously large information. To counteract this, the RV combination between chunks will throw out RVs with information greater than 10^20.

## For RV Extraction
The primary application of this software is for RV extraction from 1d time-series spectra. The underlying models are more general, but using the functions in model.py you can get models specifically for this purpose. These functions will create models with three components if pseudo-normalization is needed and two components if no normalization is needed. The other two components 


## ContainerModels
In the models.py, you will find two types of ContainerModels: Additive, and Composite.
Additive runs each of its submodels on the input, then adds the results. Composite
runs the input from each of its submodels in series.

## SubModels
In models.py, you will also find SubModels that can be used to compose some large model: Shifting,
Stretching, Convolutional, and JaxLinear. The Shifting model takes the input and adds the same value to all the elements at a given epoch. This can be used to fit for the redshift in the wavelength. Stretching multiplies all values in the input by the same number at a given epoch. And JaxLinear uses linear interpolation between the input and its control points.
