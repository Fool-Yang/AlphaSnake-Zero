from numpy import array

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, load_model, clone_model

class AlphaNNet:
    
    def __init__(self, model_name = None, input_shape = None):
        if model_name:
            self.v_net = load_model(model_name)
        elif input_shape:
            # regularization constant
            c = 1e-5
            
            X = Input(input_shape)
            
            H = Conv2D(128, (3, 3), use_bias = False, kernel_regularizer = l2(c))(X)
            H = Activation('relu')(BatchNormalization(axis = 3)(H))
            
            # a residual block
            H_shortcut = Cropping2D(cropping = 2)(H)
            H = Conv2D(128, (3, 3), use_bias = False, kernel_regularizer = l2(c))(H)
            H = Activation('relu')(BatchNormalization(axis = 3)(H))
            H = Conv2D(128, (3, 3), use_bias = False, kernel_regularizer = l2(c))(H)
            H = Activation('relu')(Add()([BatchNormalization(axis = 3)(H), H_shortcut]))
            
            H_shortcut = Cropping2D(cropping = 2)(H)
            H = Conv2D(128, (3, 3), use_bias = False, kernel_regularizer = l2(c))(H)
            H = Activation('relu')(BatchNormalization(axis = 3)(H))
            H = Conv2D(128, (3, 3), use_bias = False, kernel_regularizer = l2(c))(H)
            H = Activation('relu')(Add()([BatchNormalization(axis = 3)(H), H_shortcut]))
            
            H_shortcut = Cropping2D(cropping = 2)(H)
            H = Conv2D(128, (3, 3), use_bias = False, kernel_regularizer = l2(c))(H)
            H = Activation('relu')(BatchNormalization(axis = 3)(H))
            H = Conv2D(128, (3, 3), use_bias = False, kernel_regularizer = l2(c))(H)
            H = Activation('relu')(Add()([BatchNormalization(axis = 3)(H), H_shortcut]))
            
            H_shortcut = Cropping2D(cropping = 1)(H)
            H = Conv2D(128, (3, 3), use_bias = False, kernel_regularizer = l2(c))(H)
            H = Activation('relu')(BatchNormalization(axis = 3)(H))
            H = Conv2D(128, (3, 3), padding = "same", use_bias = False, kernel_regularizer = l2(c))(H)
            H = Activation('relu')(Add()([BatchNormalization(axis = 3)(H), H_shortcut]))
            
            H_shortcut = Cropping2D(cropping = 1)(H)
            H = Conv2D(128, (3, 3), use_bias = False, kernel_regularizer = l2(c))(H)
            H = Activation('relu')(BatchNormalization(axis = 3)(H))
            H = Conv2D(128, (3, 3), padding = "same", use_bias = False, kernel_regularizer = l2(c))(H)
            H = Activation('relu')(Add()([BatchNormalization(axis = 3)(H), H_shortcut]))
            
            H = Conv2D(2, (1, 1), use_bias = False, kernel_regularizer = l2(c))(H)
            H = Activation('relu')(BatchNormalization(axis = 3)(H))
            
            Y = Activation('tanh')(Dense(3, kernel_regularizer = l2(c))(H))
            
            self.v_net = Model(inputs = X, outputs = Y)
    
    def train(self, X, Y, epochs = 32, batch_size = 2048):
        self.v_net.fit(array(X), array(Y), epochs = epochs, batch_size = batch_size)
    
    def v(self, X):
        V = self.v_net.predict(array(X))
        center_y = len(X[0])//2
        center_x = len(X[0][0])//2
        for i in range(len(X)):
            # assign -1.0 to known obstacles
            if self.is_obstacle(X[i][center_y][center_x - 1][1]):
                V[i][0] = -1.0
            if self.is_obstacle(X[i][center_y - 1][center_x][1]):
                V[i][1] = -1.0
            if self.is_obstacle(X[i][center_y][center_x + 1][1]):
                V[i][2] = -1.0
        return V
    
    def is_obstacle(self, value):
        return value >= 0.04
    
    def copy_and_compile(self, learning_rate = 0.0001, TPU = None):
        boundaries = [20, 40, 60, 80, 100]
        values = [0.0]*(len(boundaries) + 1)
        n = learning_rate
        for i in range(len(boundaries)):
            values[i] = n
            n *= 0.25
        if TPU:
            with TPU.scope():
                nnet_copy = AlphaNNet()
                nnet_copy.v_net = clone_model(self.v_net)
                nnet_copy.v_net.build(self.v_net.layers[0].input_shape)
                nnet_copy.v_net.set_weights(self.v_net.get_weights())
                lr = schedules.PiecewiseConstantDecay(boundaries, values)
                nnet_copy.v_net.compile(
                    optimizer = Adam(learning_rate = lr),
                    loss = 'mean_squared_error'
                )
        else:
            nnet_copy = AlphaNNet()
            nnet_copy.v_net = clone_model(self.v_net)
            nnet_copy.v_net.build(self.v_net.layers[0].input_shape)
            nnet_copy.v_net.set_weights(self.v_net.get_weights())
            lr = schedules.PiecewiseConstantDecay(boundaries, values)
            nnet_copy.v_net.compile(
                optimizer = Adam(learning_rate = lr),
                loss = 'mean_squared_error'
            )
        return nnet_copy
    
    def save(self, name):
        self.v_net.save('models/' + name + '.h5')
