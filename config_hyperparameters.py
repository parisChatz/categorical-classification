from tensorflow.keras.optimizers import RMSprop

# Algorithm parameters
learning_rate = [1e-4]
momentum = 0.9

# opt = ['sgd', 'rmsprop', 'adagrad', 'adam']

batch_size = [15, 55, 105]  # todo try different. best 16
epochs = [200]
l2_score = 1e-3

model_name = 'cat_vs_dog-{}--{}--{}--{}--{}.h5'.format(learning_rate, epochs, batch_size, 'sgd', '3conv-1base')

# Dataset variables
base_dir = "images"
train_dir = base_dir + '/train'
test_dir = base_dir + '/test'
img_size = 50  # 50x50 pixels
total_train = 0
total_test = 0
