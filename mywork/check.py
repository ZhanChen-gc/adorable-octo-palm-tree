import nibabel as nib
import numpy as np

seg_path = r"D:\\predictions\\s_001.nii.gz"
seg = nib.load(seg_path).get_fdata()

print("分割数组形状：", seg.shape)
print("像素值范围：", seg.min(), "~", seg.max())
print("非零像素数量：", np.count_nonzero(seg))
print("唯一像素值：", np.unique(seg))