# Arguments for data augmentation
extreme_data_gen_args = dict(
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

minimal_data_gen_args = dict(
    rescale=1. / 255,
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    shear_range=0.1,
    brightness_range=[0.9, 1.0]  # 0.5 unchanged, 0 all black 1, all white
)

# todo remember now i do extreme data augm keep this in mind if next results are very different
