import os
import sys
import pickle
import requests
import zarr
import zipfile

# filepaths need to be identified with importlib_resources
# rather than __file__ as the latter does not work at runtime
# when the package is installed via pip install

if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files()
    # function, so use the PyPI version:
    import importlib_resources
else:
    # importlib.resources has files(), so use that:
    import importlib.resources as importlib_resources



DATASETS_PKL = [
    'MOLLI',
    'MOLLI_small',
    'MOLLI_tiny',
    'VFA',
    'VFA_small',
    'VFA_tiny',
]
DATASETS_ZARR = [
    'VFA',
    'VFA_small',
    'VFA_tiny',
    'DCE',
    'DCE_small',
    'DCE_tiny',
]


def fetch(dataset=None, clear_cache=False, download_all=False)->dict:
    """Fetch a dataset included in mdreg

    Args:
        dataset (str, optional): name of the dataset. See below for options.
        clear_cache (bool, optional): When a dataset is fetched, it is 
          downloaded and then stored in a local cache memory for faster access 
          next time it is fetched. Set clear_cache=True to delete all data 
          in the cache memory. Default is False.
        download_all (bool, optional): By default only the dataset that is 
          fetched is downloaded. Set download_all=True to download all 
          datasets at once. This will cost some time but then offers fast and 
          offline access to all datasets afterwards. This will take up around 
          300 MB of space on your hard drive. Default is False.

    Returns:
        dict: Data as a dictionary. 

    Example:
        Fetch the MOLLI images, and display as animation:

    .. plot::
        :include-source:
        :context: 

        >>> import mdreg
        >>> import mdreg.plot as plt

        Get the data:

        >>> data = mdreg.fetch('MOLLI')

        Plot as animation:

        >>> plt.animation(data['array'], vmin=0, vmax=1e4)

    Notes:

        The following datasets can be fetched: 
        
        **MOLLI**

            **Size**: 2MB

            **Background**: T1-mapping data for the kidney acquired on a 
            healthy volunteer, collected as part of the technical validation 
            efforts of the 
            `UKRIN-MAPS consortium <https://www.nottingham.ac.uk/research/groups/spmic/research/uk-renal-imaging-network/ukrin-maps.aspx>`_ 
            and the development of the 
            `UKAT package <https://github.com/UKRIN-MAPS/ukat>`_.

            **Data format**: The fetch function returns a dictionary, which 
            contains the following items: 
            
            - **array**: 4D array of signal intensities in the abdomen at 
              different inversion times.
            - **TI**: A list of inversion times in msec.

            Funding statement:

            Data collection was funded by the UKRIN-MAPS MRC Partnership 
            grant (MR/R02264X/1) and the NIHR AFiRM project (NIHR128494).

        **MOLLI_small**

            **Size**: 129KB

            **Background**: A small version of 'MOLLI' dataset, in 4x lower 
            resolution than the original. Useful for rapid testing and 
            debugging on a local machine. 

        **MOLLI_tiny**

            **Size**: 9KB

            **Background**: A tiny version of 'MOLLI' dataset, in 16x lower 
            resolution than the original. Useful for testing and debugging 
            solutions on remote machines.  

        **VFA**

            **Size**: 5MB

            **Background**: 4D variable flip angle data for T1-mapping in the 
            abdomen. Data are provided by the liver work package of 
            the `TRISTAN project <https://www.imi-tristan.eu/liver>`_  
            which develops imaging biomarkers for drug safety assessment. 
            The data and analysis was first presented at the ISMRM in 2024 
            (Min et al 2024, manuscript in press). A single set of variable 
            flip angle data are included.

            **Data format**: The fetch function returns a dictionary, which 
            contains the following items: 
            
            - **array**: 4D array of signal intensities in the liver at 
              different flip angles
            - **FA**: flip angles in degrees
            - **spacing**: voxel size in mm in x-, y-, and z-directions as an 
              array.
        
            Please reference the following abstract when using these data:

            Thazin Min, Marta Tibiletti, Paul Hockings, Aleksandra Galetin, 
            Ebony Gunwhy, Gerry Kenna, Nicola Melillo, Geoff JM Parker, 
            Gunnar Schuetz, Daniel Scotcher, John Waterton, Ian Rowe, and 
            Steven Sourbron. *Measurement of liver function with dynamic 
            gadoxetate-enhanced MRI: a validation study in healthy 
            volunteers*. Proc Intl Soc Mag Reson Med, Singapore 2024.

        **VFA_small**

            **Size**: 87KB

            **Background**: A small version of 'VFA' dataset, in 5x lower 
            resolution than the original. Useful for rapid testing and 
            debugging on a local machine. 

        **VFA_tiny**

            **Size**: 7KB

            **Background**: A tiny version of 'VFA' dataset, in 10x lower 
            resolution than the original. Useful for testing and debugging 
            cloud-based solutions that consume funds in proportion to data size. 

    """

    if dataset is None:
        v = None
    else:
        v = _fetch_dataset(dataset, '.pkl')

    if clear_cache:
        _clear_cache()

    if download_all:
        for d in DATASETS_PKL:
            _download(d, '.pkl')

    return v






def fetch_zarr(dataset=None, clear_cache=False, download_all=False)->dict:
    """Fetch a zarray dataset included in mdreg

    Args:
        dataset (str, optional): name of the dataset. See below for options.
        clear_cache (bool, optional): When a dataset is fetched, it is 
          downloaded and then stored in a local cache memory for faster access 
          next time it is fetched. Set clear_cache=True to delete all data 
          in the cache memory. Default is False.
        download_all (bool, optional): By default only the dataset that is 
          fetched is downloaded. Set download_all=True to download all 
          datasets at once. This will cost some time but then offers fast and 
          offline access to all datasets afterwards. This will take up around 
          300 MB of space on your hard drive. Default is False.

    Returns:
        zarr.Array: Data as a zarray. 

    Example:
        Fetch the MOLLI images, and display as animation:

    .. plot::
        :include-source:
        :context: 

        >>> import mdreg

        Get the data:

        >>> data = mdreg.fetch_zarr('VFA')

        Plot as animation:

        >>> mdreg.plot.animation(data, vmin=0, vmax=1e4)

    Notes:

        The following datasets can be fetched: 
        
        **MOLLI**

            **Size**: 2MB

            **Background**: T1-mapping data for the kidney acquired on a 
            healthy volunteer, collected as part of the technical validation 
            efforts of the 
            `UKRIN-MAPS consortium <https://www.nottingham.ac.uk/research/groups/spmic/research/uk-renal-imaging-network/ukrin-maps.aspx>`_ 
            and the development of the 
            `UKAT package <https://github.com/UKRIN-MAPS/ukat>`_.

            **Data format**: The fetch function returns a dictionary, which 
            contains the following items: 
            
            - **array**: 4D array of signal intensities in the abdomen at 
              different inversion times.
            - **TI**: A list of inversion times in msec.

            Funding statement:

            Data collection was funded by the UKRIN-MAPS MRC Partnership 
            grant (MR/R02264X/1) and the NIHR AFiRM project (NIHR128494).

        **MOLLI_small**

            **Size**: 129KB

            **Background**: A small version of 'MOLLI' dataset, in 4x lower 
            resolution than the original. Useful for rapid testing and 
            debugging on a local machine. 

        **VFA**

            **Size**: 5MB

            **Background**: 4D variable flip angle data for T1-mapping in the 
            abdomen. Data are provided by the liver work package of 
            the `TRISTAN project <https://www.imi-tristan.eu/liver>`_  
            which develops imaging biomarkers for drug safety assessment. 
            The data and analysis was first presented at the ISMRM in 2024 
            (Min et al 2024, manuscript in press). A single set of variable 
            flip angle data are included.

            **Data format**: The fetch function returns a dictionary, which 
            contains the following items: 
            
            - **array**: 4D array of signal intensities in the liver at 
              different flip angles
            - **FA**: flip angles in degrees
            - **spacing**: voxel size in mm in x-, y-, and z-directions as an 
              array.
        
            Please reference the following abstract when using these data:

            Thazin Min, Marta Tibiletti, Paul Hockings, Aleksandra Galetin, 
            Ebony Gunwhy, Gerry Kenna, Nicola Melillo, Geoff JM Parker, 
            Gunnar Schuetz, Daniel Scotcher, John Waterton, Ian Rowe, and 
            Steven Sourbron. *Measurement of liver function with dynamic 
            gadoxetate-enhanced MRI: a validation study in healthy 
            volunteers*. Proc Intl Soc Mag Reson Med, Singapore 2024.

        **VFA_small**

            **Size**: 87KB

            **Background**: A small version of 'VFA' dataset, in 5x lower 
            resolution than the original. Useful for rapid testing and 
            debugging on a local machine. 

        **VFA_tiny**

            **Size**: 7KB

            **Background**: A tiny version of 'VFA' dataset, in 10x lower 
            resolution than the original. Useful for testing and debugging 
            cloud-based solutions that consume funds in proportion to data size. 

    """

    if dataset is None:
        v = None
    else:
        v = _fetch_dataset(dataset, '.zip')

    if clear_cache:
        _clear_cache()

    if download_all:
        for d in DATASETS_PKL:
            _download(d, '.zip')

    return v


def _clear_cache():
    """
    Clear the folder where the data downloaded via fetch are saved.

    Note if you clear the cache the data will need to be downloaded again 
    if you need them.
    """

    f = importlib_resources.files('mdreg.datafiles')
    for item in f.iterdir(): 
        if item.is_file(): 
            item.unlink() # Delete the file


def _fetch_dataset(dataset, ext):

    f = importlib_resources.files('mdreg.datafiles')
    datafile = str(f.joinpath(dataset + ext))

    # If this is the first time the data are accessed, download them.
    if not os.path.exists(datafile):
        _download(dataset, ext)

    if ext=='.pkl':
        with open(datafile, 'rb') as f:
            return pickle.load(f)
        
    if ext=='.zip':
        extracted = os.path.join(os.path.dirname(datafile), dataset)
        if not os.path.exists(extracted):
            with zipfile.ZipFile(datafile, 'r') as z:
                z.extractall(extracted)
        return zarr.open(extracted, mode='r')


def _download(dataset, ext):
        
    f = importlib_resources.files('mdreg.datafiles')
    datafile = str(f.joinpath(dataset + ext))

    if os.path.exists(datafile):
        return

    # Dataset location
    version_doi = "14933756" # This will change if a new version is created on zenodo
    file_url = "https://zenodo.org/records/" + version_doi + "/files/" + dataset + ext

    # Make the request and check for connection error
    try:
        file_response = requests.get(file_url) 
    except requests.exceptions.ConnectionError as err:
        raise requests.exceptions.ConnectionError(
            "\n\n"
            "A connection error occurred trying to download the test data \n"
            "from Zenodo. This usually happens if you are offline. The \n"
            "first time a dataset is fetched via mdreg.fetch you need to \n"
            "be online so the data can be downloaded. After the first \n"
            "time they are saved locally so afterwards you can fetch \n"
            "them even if you are offline. \n\n"
            "The detailed error message is here: " + str(err)) 
    
    # Check for other errors
    file_response.raise_for_status()

    # Save the file locally 
    with open(datafile, 'wb') as f:
        f.write(file_response.content)