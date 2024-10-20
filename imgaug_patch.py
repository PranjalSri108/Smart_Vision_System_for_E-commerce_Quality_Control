import numpy as np

# Define the types manually since np.sctypes is removed in NumPy 2.0
NP_FLOAT_TYPES = {np.float16, np.float32, np.float64}
NP_INT_TYPES = {np.int8, np.int16, np.int32, np.int64}
NP_UINT_TYPES = {np.uint8, np.uint16, np.uint32, np.uint64}

# Monkey patch imgaug
def patch_imgaug():
    import imgaug.imgaug
    imgaug.imgaug.NP_FLOAT_TYPES = NP_FLOAT_TYPES
    imgaug.imgaug.NP_INT_TYPES = NP_INT_TYPES
    imgaug.imgaug.NP_UINT_TYPES = NP_UINT_TYPES
