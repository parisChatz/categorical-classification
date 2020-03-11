from tensorflow.keras.optimizers import RMSprop

# maybe regularization? no overfit so dont need
# todo maybe BatchNormalization()?
# todo transfer learning

# # Algorithm parameters
learning_rates = [1e-2]
momentums = [0.9]

models = ['vgg1', 'vgg2', 'vgg3', 'lenet5']

optimizers = ['SGD', 'Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Nadam', 'RMSprop']

best_test_accuracy = 0

# reminder 16 batch
batch_size = [16]
epochs = [400]

# l2_score = [0.01, 0.001, 0.0001]
l2_score = [0]

# Dataset variables
color = ['grayscale', 'rgb']
base_dir = "images"
train_dir = base_dir + '/train'
val_dir = base_dir + '/val'
test_dir = base_dir + '/test'

img_size = 50  # 50x50 pixels
total_train = 0
total_test = 0
