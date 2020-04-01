from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model, load_model, clone_model


class AlphaNNet:
    
    def __init__(self, model = None, ins = None):
        if model:
            self.v_net = load_model(model)
        elif ins:
            X = Input(ins)
            H = BatchNormalization(axis=3)(X)
            
            H_shortcut = BatchNormalization(axis=3)(Conv2D(64, (7, 7), use_bias=False)(H))
            H = Activation('relu')(BatchNormalization(axis=3)(Conv2D(64, (3, 3), use_bias=False)(H)))
            H = Activation('relu')(BatchNormalization(axis=3)(Conv2D(64, (3, 3), use_bias=False)(H)))
            H = BatchNormalization(axis=3)(Conv2D(64, (3, 3), use_bias=False)(H))
            H = Activation('relu')(Add()([H, H_shortcut]))
            
            H_shortcut = BatchNormalization(axis=3)(Conv2D(64, (7, 7), use_bias=False)(H))
            H = Activation('relu')(BatchNormalization(axis=3)(Conv2D(64, (3, 3), use_bias=False)(H)))
            H = Activation('relu')(BatchNormalization(axis=3)(Conv2D(64, (3, 3), use_bias=False)(H)))
            H = BatchNormalization(axis=3)(Conv2D(64, (3, 3), use_bias=False)(H))
            H = Activation('relu')(Add()([H, H_shortcut]))
            
            H_shortcut = BatchNormalization(axis=3)(Conv2D(64, (7, 7), use_bias=False)(H))
            H = Activation('relu')(BatchNormalization(axis=3)(Conv2D(64, (3, 3), use_bias=False)(H)))
            H = Activation('relu')(BatchNormalization(axis=3)(Conv2D(64, (3, 3), use_bias=False)(H)))
            H = BatchNormalization(axis=3)(Conv2D(64, (3, 3), use_bias=False)(H))
            H = Activation('relu')(Add()([H, H_shortcut]))
            
            H = Activation('relu')(BatchNormalization(axis=3)(Conv2D(64, (3, 3), use_bias=False)(H)))
            
            Y = Activation('tanh')(Dense(3)(Flatten()(H)))
            
            self.v_net = Model(inputs = X, outputs = Y)
    
    def train(self, X, Y, ep = None, bs = None):
        self.v_net.fit(X, Y, epochs = ep, batch_size = bs)
    
    def v(self, X):
        return self.v_net.predict(X)
    
    def copy(self, lr = 0.001):
        nnet_copy = AlphaNNet()
        # value
        nnet_copy.v_net = clone_model(self.v_net)
        nnet_copy.v_net.build(self.v_net.layers[0].input_shape)
        nnet_copy.v_net.set_weights(self.v_net.get_weights())
        nnet_copy.v_net.compile(
            optimizer = Adam(learning_rate = lr),
            loss = "mean_squared_error"
        )
        return nnet_copy
    
    def save(self, name):
        self.v_net.save('models/' + name + '.h5')
