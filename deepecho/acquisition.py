# This file is part of tdsi-deep-echo-challenge

import SimpleITK as sitk
import numpy as np


def load_mhd_data(filename):
    """
    This function reads in a .mhd file, and returns the image (pixel values contained in the associated .raw file),
    along with some meta-info.
    :param filename: path to the source file (.mhd)
    :return: the image array (as a np.ndarray), the origin, the spacing and the dimensions (meta-info)
    """
    # reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)
    # onvert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    image = sitk.GetArrayFromImage(itkimage)
    # if the numpy.ndarray is of shape (1,:,:), convert it to a 2D-array
    z_len, _, _ = image.shape
    if z_len == 1:
        image = image[z_len-1, :, :]
    # read the origin of the image frame
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    # ead the spacing along each dimension (pixel widths)
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    # ead the dimensions
    dimensions = np.array(list(reversed(itkimage.GetSize())))

    return image, origin, spacing, dimensions

