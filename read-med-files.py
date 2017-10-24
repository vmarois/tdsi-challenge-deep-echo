import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def load_mhd_raw_data(filename):
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


# Read 'End of Diastole' image & mask
image_ed, _, _, _ = load_mhd_raw_data('data/patient0001/patient0001_4CH_ED.mhd')
image__ed_gt, _, _, _ = load_mhd_raw_data('data/patient0001/patient0001_4CH_ED_gt.mhd')
"""
fig_ed = plt.figure()
ax1_ed = fig_ed.add_subplot(1, 2, 1)
ax1_ed.imshow(image_ed, cmap='Greys')
ax1_ed.set_title('ED image')
ax2_ed = fig_ed.add_subplot(1, 2, 2)
ax2_ed.imshow(image__ed_gt, cmap='Greys')
ax2_ed.set_title('ED Ground Truth Mask')
"""
# Read 'End of Systole' image & mask
image_es, _, _, _ = load_mhd_raw_data('data/patient0001/patient0001_4CH_ES.mhd')
image__es_gt, _, _, _ = load_mhd_raw_data('data/patient0001/patient0001_4CH_ES_gt.mhd')
"""
fig_es = plt.figure()
ax1_es = fig_es.add_subplot(1, 2, 1)
ax1_es.imshow(image_es, cmap='Greys')
ax1_es.set_title('ES image')
ax2_es = fig_es.add_subplot(1, 2, 2)
ax2_es.imshow(image__es_gt, cmap='Greys')
ax2_es.set_title('ES Ground Truth Mask')

# Read 'sector' image
image_sector, _, _, _ = load_mhd_raw_data('data/patient0001/patient0001_4CH_sector.mhd')
fig_sector = plt.figure()
ax_sector = fig_sector.add_subplot(1, 1, 1)
ax_sector.imshow(image_sector, cmap='Greys')
ax_sector.set_title('Sector image')
"""
# Plot the 3 figures
# plt.show()


# Keep only the region corresponding to the left ventricle and plot them
vent_ed = (image__ed_gt == 1)[:, :]
vent_es = (image__es_gt == 1)[:, :]
"""
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(vent_ed, cmap='Greys')
ax1.set_title('ED mask of ventricle')
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(vent_es, cmap='Greys')
ax2.set_title('ES mask of ventricle')
plt.show()
"""


def findCenter(img):
    """
    This function returns the center coordinates of the different connected regions of an image.
    :param img: input image
    :return: ([x1, x2 ... xn], [y1, y2 ... yn]) where xi,yi are the coordinates of the ith region detected in the
    image (total of n regions). If only one region is detected, the 2 coordinates are returned as a tuple (x,y).
    """
    # use a boolean condition to find where pixel values are = 1
    blobs = (img == 1)
    # label the n connected regions that satisfy this condition
    labels, nlabels = ndimage.label(blobs)
    # Find their unweighted centroids
    r, c = np.vstack(ndimage.center_of_mass(blobs, labels, np.arange(nlabels) + 1)).T  # returned as np.ndarray
    # round the values to int (since pixel coordinates)
    r = np.round(r).astype(int)
    c = np.round(c).astype(int)
    if nlabels == 1:
        return r[0], c[0]
    else:
        return r.tolist(), c.tolist()


# Get the center of the left ventricle, for ED & ES
r_ed, c_ed = findCenter(vent_ed)
vent_ed[r_ed-2:r_ed+2, c_ed-2:c_ed+2] = 0  # Enlarge a bit the center for better visualization

r_es, c_es = findCenter(vent_es)
vent_es[r_es-2:r_es+2, c_es-2:c_es+2] = 0  # Enlarge a bit the center for better visualization
"""
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(vent_ed, cmap='Greys')
ax1.set_title('Center of left ventricle (ED)')
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(vent_es, cmap='Greys')
ax2.set_title('Center of left ventricle (ES)')

plt.show()
"""


def findMainOrientation(img, pixelvalue):
    """
    This function returns the main orientation of the region composed of pixels of the specified value.
    :param img: input image, pixelvalue: the value used to filter the pixels
    :return: the x- & y-eigenvalues of the region as a tuple (correspond to the main orientation, see
    https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/)
    """
    # get the indices of the pixels of value equal to pixelvalue
    y, x = np.where(img == pixelvalue)
    #  subtract mean from each dimension.
    x = x - np.mean(x)
    y = y - np.mean(y)
    coords = np.vstack([x, y])
    # covariance matrix and its eigenvectors and eigenvalues
    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)
    # sort eigenvalues in decreasing order
    sort_indices = np.argsort(evals)[::-1]
    evec1, _ = evecs[:, sort_indices]
    # eigenvector with largest eigenvalue
    x_v1, y_v1 = evec1
    return x_v1, y_v1


# Get the orientation of the left ventricle during ED
x_v1, y_v1 = findMainOrientation(vent_ed, 1)
"""
# Plot it on top of the left ventricle region
y, x = np.nonzero(vent_ed)
scale = 260
plt.plot(x, y, color='black')
plt.plot([c_ed - x_v1*scale, c_ed + x_v1*scale], [r_ed - y_v1*scale, r_ed + y_v1*scale], color='red')
plt.axis('equal')
plt.gca().invert_yaxis()  # Match the image system with origin at top left
plt.title('Main orientation of the left ventricle during ED')
plt.show()
"""