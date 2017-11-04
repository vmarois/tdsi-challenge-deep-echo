# This file is part of tdsi-deep-echo-challenge
import sys

# Set python path to find the local deepecho package
sys.path.insert(0, 'pwd')
from deepecho import *


# Read 'End of Diastole' image & mask
image_ed, _, _, _ = load_mhd_data('data/patient0001/patient0001_4CH_ED.mhd')
image_ed_gt, _, _, _ = load_mhd_data('data/patient0001/patient0001_4CH_ED_gt.mhd')
#plotImageMask(image_ed, image_ed_gt)

# Read 'End of Systole' image & mask
image_es, _, _, _ = load_mhd_data('data/patient0001/patient0001_4CH_ES.mhd')
image_es_gt, _, _, _ = load_mhd_data('data/patient0001/patient0001_4CH_ES_gt.mhd')
#plotImageMask(image_ed, image_ed_gt, phase='ES')

# Keep only the region corresponding to the left ventricle
vent_ed = getRoi(image_ed_gt, 1)
vent_es = getRoi(image_es_gt, 1)

# Get the center of the left ventricle, for ED & ES
r_ed, c_ed = findCenter(vent_ed)
r_es, c_es = findCenter(vent_es)

# Get the orientation of the left ventricle during ED
x_v1, y_v1 = findMainOrientation(vent_ed, 1)

#plotCenterOrientation(vent_ed, (r_ed, c_ed), (x_v1, y_v1))

# try to create a panda DF containing center & orientation info for left ventricle (ED) for all patients
dfED = createDataFrame('data')
print('DataFrame containing center & orientation for ED :\n', dfED.head())
print('\n')
dfES = createDataFrame('data', phase='ES')
print('DataFrame containing center & orientation for ES :\n', dfES.head())

