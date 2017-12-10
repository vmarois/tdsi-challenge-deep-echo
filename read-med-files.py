# This file is part of tdsi-deep-echo-challenge
import sys

# Set python path to find the local deepecho package
sys.path.insert(0, 'pwd')
from deepecho import *


# Read 'End of Diastole' image & mask
image_ed, _, _, _ = load_mhd_data('data/patient0001/patient0001_4CH_ED.mhd')
image_ed_gt, _, _, _ = load_mhd_data('data/patient0001/patient0001_4CH_ED_gt.mhd')

# Optionally, plot them side-by-side
plotImageMask(image_ed, image_ed_gt)

# Read 'End of Systole' image & mask
image_es, _, _, _ = load_mhd_data('data/patient0001/patient0001_4CH_ES.mhd')
image_es_gt, _, _, _ = load_mhd_data('data/patient0001/patient0001_4CH_ES_gt.mhd')

# Optionally, plot them side-by-side
#plotImageMask(image_es, image_es_gt, phase='ES')

# Keep only the region of interest corresponding to the left ventricle (on the mask)
vent_ed = getRoi(image_ed_gt, 1)
vent_es = getRoi(image_es_gt, 1)

# Resize the images & masks to 96 x 96 to keep the input size of the network manageable
image_ed_resized, vent_ed_resized = resizeImgArr(image_ed, vent_ed)
image_es_resized, vent_es_resized = resizeImgArr(image_es, vent_es)

# Get the center of the left ventricle, for ED & ES
r_ed, c_ed = findCenter(vent_ed_resized)
r_es, c_es = findCenter(vent_es_resized)

# Get the orientation of the left ventricle during ED
x_v1, y_v1 = findMainOrientation(vent_ed_resized, 1)

plotCenterOrientation(vent_ed_resized, (r_ed, c_ed), (x_v1, y_v1))
