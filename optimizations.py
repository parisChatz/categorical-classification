from matplotlib import pyplot


# best pipeline? : best architecture -> best optimizer -> best LR -> best momentum TODO yes!
# Best optimizer
def find_optimizer(trainX, trainy, testX, testy):
    # create learning curves for different optimizers
    optimizer = ['sgd', 'rmsprop', 'adagrad', 'adam']
    for i in range(len(optimizer)):
        # determine the plot number
        plot_no = 220 + (i + 1)
        pyplot.subplot(plot_no)
        # fit Model and plot learning curves for an optimizer
        # fit_model(trainX, trainy, testX, testy, optimizer[i])
    # show learning curves
    pyplot.show()


# Best learning rate
def find_learning_rate(trainX, trainy, testX, testy):
    # create learning curves for different learning rates
    learning_rates = [1E-0, 1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6, 1E-7]
    for i in range(len(learning_rates)):
        # determine the plot number
        plot_no = 420 + (i + 1)
        pyplot.subplot(plot_no)
        # fit Model and plot learning curves for a learning rate
        # fit_model(trainX, trainy, testX, testy, learning_rates[i])
    # show learning curves
    pyplot.show()


# Best momentum
def find_momentum(trainX, trainy, testX, testy):
    # create learning curves for different momentums
    momentums = [0.0, 0.5, 0.9, 0.99]
    for i in range(len(momentums)):
        # determine the plot number
        plot_no = 220 + (i + 1)
        pyplot.subplot(plot_no)
        # fit Model and plot learning curves for a momentum
        # fit_model(trainX, trainy, testX, testy, momentums[i])
    # show learning curves
    pyplot.show()
