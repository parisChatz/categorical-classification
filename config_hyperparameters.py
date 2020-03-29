from tensorflow.keras.optimizers import RMSprop

# maybe regularization? no overfit so dont need
# todo maybe BatchNormalization()?
# todo transfer learning


save_everything = False

# # Algorithm parameters
learning_rates = [1e-2]

models = ['vgg2']

# optimizers = ['SGD', 'Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Nadam', 'RMSprop']
optimizers = ['Adam', 'RMSprop']

best_test_accuracy = 0

# reminder 16 batch
batch_size = [600]
epochs = [5]

# l2_score = [0.01, 0.001, 0.0001]
l2_score = [0]

# Dataset variables
color = ['grayscale', 'rgb']
base_dir = "images"
train_dir = base_dir + '/train'
val_dir = base_dir + '/val'
test_dir = base_dir + '/test'
plot_path = "documentation/optimizers2"
img_size = 50  # 50x50 pixels
total_train = 0
total_test = 0
