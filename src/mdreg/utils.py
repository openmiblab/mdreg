import numpy as np


def defo_jacobian_2d(defo):
    """
    Calculate the Jacobian matrix and determinant from a 2D deformation field.
    Can process multi-slice images, but the actual deformation 
    field/registration must be 2D.
    
    Parameters
    ----------
    defo : np.ndarray
        The deformation field to calculate the Jacobian from.
        Dimensions are expected in the order [x, y, z, t, d], where x, y, z are 
        the spatial dimensions, t is the time/dynamic, and d is the dimension 
        of the deformation field vector (two for 2D registration).

    Returns
    -------
    jac_mat : np.ndarray
        The Jacobian matrix of the deformation field with dimensions [x, y, z, t]
    jac_det : np.ndarray
        The determinant of the Jacobian matrix.
    """
    if defo.ndim != 5:
        raise ValueError('Deformation field must have dimensions '
                         '[x, y, z, t, d].')
    if defo.shape[-1] != 2:
        raise ValueError('Deformation field must be 2D.')

    jac_mat = np.zeros((defo.shape[0], defo.shape[1], defo.shape[2], 
                        defo.shape[3], 2, 2))
    jac_det = np.zeros((defo.shape[:4]))

    for t in range(defo.shape[3]):
        for z in range(defo.shape[2]):
            grad_xx, grad_xy = np.gradient(defo[:, :, z, t, 1])
            grad_yx, grad_yy = np.gradient(defo[:, :, z, t, 0])

            grad_xx += 1
            grad_yy += 1
            jac_mat[:, :, z, t, 0, 0] = grad_xx
            jac_mat[:, :, z, t, 0, 1] = grad_xy
            jac_mat[:, :, z, t, 1, 0] = grad_yx
            jac_mat[:, :, z, t, 1, 1] = grad_yy

            jac_det[:, :, z, t] = np.linalg.det(jac_mat[:, :, z, t, :, :])

    return jac_mat, jac_det


def defo_norm(defo, norm='euclidian'):
    """
    Calculate the norm of a deformation field.
    
    Parameters
    ----------
    defo : np.ndarray
        The deformation field to calculate the norm from. 
        Dimensions are expected in the order [x, y, z, t, d], where x, y, z are 
        the spatial dimensions, t is the time/dynamic, and d is the dimension 
        of the deformation field (two for 2D registration, 3 for 3D registration)
    norm : str
        The type of norm to use. Options are 'euclidian', 'max' or 'eumip'
        The latter is the maximum projection over time of the euclidian norm.
        Default is 'euclidian'.

    Returns
    -------
    norm : np.ndarray
        The norm of the deformation field with dimensions [x, y, z, t] or 
        [x,y,z] (for option 'eumip')
    """
    if norm == 'euclidian':
        return np.linalg.norm(defo, axis=-1)
    elif norm == 'max':
        return np.amax(defo, axis=-1)
    elif norm == 'eumip':
        return np.amax(np.linalg.norm(defo, axis=-1), axis=-1)
    else:
        raise ValueError('Norm ' + str(norm) + ' is not available.')
    
