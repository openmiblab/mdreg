import os
import numpy as np
import zarr
import math
import inspect
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import ArtistAnimation
except:
    not_installed = True
else:
    not_installed = False


def animation(
        array, 
        path=None, 
        filename='animation', 
        vmin=None, 
        vmax=None, 
        title = '', 
        interval=250, 
        show=True,
        verbose=0,
    ):

    """
    Produce an animation of a 3D image.

    Parameters
    ----------
    array : numpy.array | zarr.Array
        The 3D image to animate.
    path : str, optional
        The path to save the animation. The default is None.
    filename : str, optional
        The filename of the animation. The default is 'animation'.
    vmin : float, optional
        The minimum value for the colormap. The default is None.
    vmax : float, optional
        The maximum value for the colormap. The default is None.
    title : str, optional
        The title of the animation to be rendered on the figure.
        The default is ''.
    interval : int, optional
        The interval between frames. The default is 250ms.
    show : bool, optional
        Whether to display the animation. The default is False.
    verbose : int, optional
        Set to 1 to show progress bars. Default is 0 (no progress bars).
    """

    if not_installed:
        raise ImportError(
            "The plot functions are optional - please install mdreg as "
            "pip install mdreg[plot] if you want to use these."
        )

    shape = np.shape(array)
    titlesize = 10

    if array.ndim == 4: ##save 3D data

        # Determine the grid size for the panels
        num_slices = array.shape[2]
        grid_size = math.ceil(math.sqrt(num_slices))

        fig_3d, axes1 = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))
        fig_3d.subplots_adjust(wspace=0.5, hspace=0.01)

        fig_3d.suptitle('{} \n \n'.format(title), fontsize=titlesize+2)
        plt.tight_layout()

        for i in range(grid_size * grid_size):
                row = i // grid_size
                col = i % grid_size
                if i < num_slices:
                    im = np.nan_to_num(array[:, :, i, 0])
                    axes1[row, col].imshow(im.T, cmap='gray', animated=True, vmin=vmin, vmax=vmax)
                    axes1[row, col].set_title(f'Slice {i+1}', fontsize=titlesize)
                else:
                    axes1[row, col].axis('off')  # Turn off unused subplots
                axes1[row, col].set_xticks([])  # Remove x-axis ticks
                axes1[row, col].set_yticks([])

        images = []
        for j in tqdm(range(array.shape[-1]), desc='Building animation',  disable=verbose==0):
            ims = []
            for i in range(grid_size * grid_size):
                row = i // grid_size
                col = i % grid_size
                if i < num_slices:
                    im = np.nan_to_num(array[:, :, i, j])
                    im = axes1[row, col].imshow(im.T, cmap='gray', animated=True, vmin=vmin, vmax=vmax)
                    ims.append(im)
            images.append(ims,)

        anim = ArtistAnimation(fig_3d, images, interval=interval, repeat_delay=interval)
        if path is not None:
            file_3D_save = os.path.join(path, filename)
            anim.save(file_3D_save + "_"  + ".gif")
        if show:
            plt.show()
            return anim
        else:
            plt.close()
            return

    else: # save 2D data  
        fig, ax = plt.subplots()
        im = np.nan_to_num(array[:, :, 0])
        im = ax.imshow(im.T, cmap='gray', animated=True, vmin=vmin, vmax=vmax)
        ims = []
        for i in tqdm(range(shape[-1]), desc='Building animation',  disable=verbose==0):
            im = np.nan_to_num(array[:, :, i])
            im = ax.imshow(im.T, cmap='gray', animated=True, vmin=vmin, vmax=vmax) 
            ims.append([im]) 
        anim = ArtistAnimation(fig, ims, interval=interval)
        if path is not None:
            file_3D_save = os.path.join(path, filename)
            anim.save(file_3D_save + ".gif")
        if show:
            plt.show()
            return anim
        else:
            plt.close()
            return


def series(moving, fixed, coreg, path=None, filename='animation', 
                vmin=None, vmax=None, interval=250, show=True):

    """
    Produce an animation of the original, fitted and coregistered images.

    Parameters
    ----------
    moving : numpy.array
        The moving image.
    fixed : numpy.array
        The fixed/fitted image.
    coreg : numpy.array
        The coregistered image.
    path : str, optional
        The path to save the animation. The default is None.
    filename : str, optional
        The filename of the animation. The default is 'animation'.
    vmin : float, optional
        The minimum value for the colormap. The default is None.
    vmax : float, optional
        The maximum value for the colormap. The default is None.
    interval : int, optional
        The interval between frames. The default is 250ms.
    show : bool, optional
        Whether to display the animation. The default is False.

    """

    if not_installed:
        raise ImportError(
            "The plot functions are optional - please install mdreg as "
            "pip install mdreg[plot] if you want to use these."
        )

    titlesize = 6

    if (moving.ndim == fixed.ndim == coreg.ndim == 4):

        # Determine the grid size for the panels
        num_slices = moving.shape[2]
        grid_size = math.ceil(math.sqrt(num_slices))
        titles = ['Original Data', 'Model Fit', 'Coregistered']
        anims = []

        for data in (moving, fixed, coreg):
            fig_3d, axes1 = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))
            fig_3d.subplots_adjust(wspace=0.5, hspace=0.01)
            data_name = titles[[np.array_equal(data, moving), np.array_equal(data, fixed), np.array_equal(data, coreg)].index(True)]

            fig_3d.suptitle('Series Type: {} \n \n'.format(data_name), fontsize=titlesize+2)
            plt.tight_layout()

            for i in range(grid_size * grid_size):
                    row = i // grid_size
                    col = i % grid_size
                    if i < num_slices:
                        axes1[row, col].imshow(data[:, :, i, 0].T, cmap='gray', animated=True, vmin=vmin, vmax=vmax)
                        axes1[row, col].set_title('Slice {}'.format(i+1), fontsize=titlesize)
                    else:
                        axes1[row, col].axis('off')  # Turn off unused subplots
                    axes1[row, col].set_xticks([])  # Remove x-axis ticks
                    axes1[row, col].set_yticks([])

            images = []
            for j in range(data.shape[-1]):
                ims = []
                for i in range(grid_size * grid_size):
                    row = i // grid_size
                    col = i % grid_size
                    if i < num_slices:
                        im = axes1[row, col].imshow(data[:, :, i, j].T, cmap='gray', animated=True, vmin=vmin, vmax=vmax)
                        ims.append(im)
                images.append(ims,)

            anim = ArtistAnimation(fig_3d, images, interval=interval, repeat_delay=interval)
            if path is not None:
                file_3D_save = os.path.join(path, filename)
                data_type_name = _get_var_name(data)
                anim.save(file_3D_save + "_" + data_type_name + ".gif")
            if show:
                plt.show()
                anims.append(anim)
            else:
                plt.close()
        if show:
            return anims
        else:
            return

    elif not (moving.ndim == fixed.ndim == coreg.ndim):
        raise ValueError('Dimension mismatch in arrays provided. Please '
                         'ensure the three arrays have the same dimensions')

    fig, ax = plt.subplots(figsize=(6, 2), ncols=3, nrows=1)
    ax[0].set_title('Model fit', fontsize=titlesize+2)
    ax[1].set_title('Original Data', fontsize=titlesize+2)
    ax[2].set_title('Coregistered', fontsize=titlesize+2)
    for i in range(3):
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
    ax[0].imshow(fixed[:,:,0].T, cmap='gray', animated=True, vmin=vmin, vmax=vmax)
    ax[1].imshow(moving[:,:,0].T, cmap='gray', animated=True, vmin=vmin, vmax=vmax) 
    ax[2].imshow(coreg[:,:,0].T, cmap='gray', animated=True, vmin=vmin, vmax=vmax) 
    ims = []
    for i in range(fixed.shape[-1]):
        im0 = ax[0].imshow(fixed[:,:,i].T, cmap='gray', animated=True, vmin=vmin, vmax=vmax)
        im1 = ax[1].imshow(moving[:,:,i].T, cmap='gray', animated=True, vmin=vmin, vmax=vmax) 
        im2 = ax[2].imshow(coreg[:,:,i].T, cmap='gray', animated=True, vmin=vmin, vmax=vmax)  
        ims.append([im0, im1, im2]) 
    anim = ArtistAnimation(fig, ims, interval=interval, repeat_delay=interval)
    if path is not None:
        file_3D_save = os.path.join(path, filename)
        anim.save(file_3D_save + ".gif")
    if show:
        plt.show()  
        return anim 
    else:
        plt.close()
        return
    

def par(array:np.ndarray, path=None, filename='image', vmin=None, vmax=None, 
             title='', show=True):

    """
    Plot a 2D or 3D image.

    Parameters
    ----------
    array : numpy.array
        The 2D or 3D image to animate.
    path : str, optional
        The path to save the animation. The default is None.
    filename : str, optional
        The filename to save the image. The default is 'image'.
    vmin : float, optional
        The minimum value for the colormap. The default is None.
    vmax : float, optional
        The maximum value for the colormap. The default is None.
    title : str, optional
        The title of the plot. The default is ''.
    show : bool, optional
        Whether to display the animation. The default is False.

    """
    
    if not_installed:
        raise ImportError(
            "The plot functions are optional - please install mdreg as "
            "pip install mdreg[plot] if you want to use these."
        )
    
    if array.ndim not in [2,3]:
        raise ValueError("Parameter array must be 2D or 3D")

    titlesize = 10

    if array.ndim == 3: ##save 3D data

        # Determine the grid size for the panels
        num_slices = array.shape[2]
        grid_size = math.ceil(math.sqrt(num_slices))

        fig_3d, axes1 = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))
        fig_3d.subplots_adjust(wspace=0.5, hspace=0.01)

        fig_3d.suptitle('{} \n \n'.format(title), fontsize=titlesize+2)
        plt.tight_layout()

        for i in range(grid_size * grid_size):
            row = i // grid_size
            col = i % grid_size
            if i < num_slices:
                axes1[row, col].imshow(np.nan_to_num(array[:, :, i]).T, cmap='gray', 
                                        vmin=vmin, vmax=vmax)
                axes1[row, col].set_title('Slice {}'.format(i+1), 
                                          fontsize=titlesize)
            else:
                axes1[row, col].axis('off')  # Turn off unused subplots
            axes1[row, col].set_xticks([])  # Remove x-axis ticks
            axes1[row, col].set_yticks([])

        if path is not None:
            file_3D_save = os.path.join(path, filename)
            fig_3d.save(file_3D_save + ".gif")
        if show:
            plt.show()
            return fig_3d
        else:
            plt.close()
            return

    else: # save 2D data  
        fig, ax = plt.subplots()
        im = ax.imshow(np.nan_to_num(array[:,:]).T, cmap='gray', vmin=vmin, vmax=vmax)
        if path is not None:
            file_3D_save = os.path.join(path, filename)
            fig.save(file_3D_save + ".gif")
        if show:
            plt.show()
            return fig
        else:
            plt.close()
            return




def _get_var_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    for var_name, var_val in callers_local_vars:
        if var_val is var:
            return var_name
    return None
