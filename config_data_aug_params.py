# Arguments for data augmentation
data_gen_args = dict(
    rescale=1. / 255,
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    brightness_range=[0.8, 1.0]  # 0.5 unchanged, 0 all black 1, all white
)
