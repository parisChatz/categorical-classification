from tensorflow.keras.optimizers import RMSprop

# todo maybe regularization?
# todo maybe BatchNormalization()?
# todo transfer learning

# Algorithm parameters
# learning_rates = [1e-2, 1e-3, 1e-4]
# momentum = 0.9

models = ['vgg1', 'vgg2', 'vgg3', 'lenet5']
optimizers = ['SGD', 'AdaDelta', 'AdaGrad', 'Adam']

# reminder 60 batch is too few update steps for weight
batch_size = [16]
epochs = [300]

# reminder 1e-5 is to small
# l2_score = [0.01, 0.001, 0.0001]
l2_score = [0]
# model_name = 'cat_vs_dog-{}--{}--{}--{}--{}.h5'.format(learning_rate, epochs, batch_size, 'sgd', '3conv-1base')

# Dataset variables
base_dir = "images"
train_dir = base_dir + '/train'
test_dir = base_dir + '/test'
img_size = 50  # 50x50 pixels
total_train = 0
total_test = 0
