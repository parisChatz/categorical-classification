# Arguments for data augmentation
data_gen_args = dict(
    rescale=1. / 255,
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=80,
    width_shift_range=0.4,
    height_shift_range=0.4,
    zoom_range=0.4,
    horizontal_flip=True,
    shear_range=0.4,
    brightness_range=[0.4, 1.0]  # 0.5 unchanged, 0 all black 1, all white
)
# todo remember now i do extreme data augm keep this in mind if next results are very different
