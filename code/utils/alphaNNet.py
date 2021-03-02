from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model, load_model, clone_model

class AlphaNNet:
    
    def __init__(self, model = None, ins = None):
        if model:
            self.v_net = load_model(model)
        elif ins:
            X = Input(ins)
            
            H = Activation('relu')(BatchNormalization(axis=3)(Conv2D(128, (3, 3), use_bias=False)(X)))
            
            H_shortcut = Cropping2D(cropping=2)(H)
            H = Activation('relu')(BatchNormalization(axis=3)(Conv2D(128, (3, 3), use_bias=False)(H)))
            H = BatchNormalization(axis=3)(Conv2D(128, (3, 3), use_bias=False)(H))
            H = Activation('relu')(Add()([H, H_shortcut]))
            
            H_shortcut = Cropping2D(cropping=2)(H)
            H = Activation('relu')(BatchNormalization(axis=3)(Conv2D(128, (3, 3), use_bias=False)(H)))
            H = BatchNormalization(axis=3)(Conv2D(128, (3, 3), use_bias=False)(H))
            H = Activation('relu')(Add()([H, H_shortcut]))
            
            H_shortcut = Cropping2D(cropping=2)(H)
            H = Activation('relu')(BatchNormalization(axis=3)(Conv2D(128, (3, 3), use_bias=False)(H)))
            H = BatchNormalization(axis=3)(Conv2D(128, (3, 3), use_bias=False)(H))
            H = Activation('relu')(Add()([H, H_shortcut]))
            
            H_shortcut = Cropping2D(cropping=2)(H)
            H = Activation('relu')(BatchNormalization(axis=3)(Conv2D(128, (3, 3), use_bias=False)(H)))
            H = BatchNormalization(axis=3)(Conv2D(128, (3, 3), use_bias=False)(H))
            H = Activation('relu')(Add()([H, H_shortcut]))
            
            Y = Activation('sigmoid')(Dense(3)(Flatten()(H)))
            
            self.v_net = Model(inputs = X, outputs = Y)
    
    def train(self, X, Y, ep = None, bs = None):
        self.v_net.fit(X, Y, epochs = ep, batch_size = bs)
    
    def v(self, X):
        return self.v_net.predict(X)
    
    def copy(self):
        nnet_copy = AlphaNNet()
        # value
        nnet_copy.v_net = clone_model(self.v_net)
        nnet_copy.v_net.build(self.v_net.layers[0].input_shape)
        nnet_copy.v_net.set_weights(self.v_net.get_weights())
        boundaries = [20, 40]
        values = [0.0001, 0.00005, 0.00002]
        lr = schedules.PiecewiseConstantDecay(boundaries, values)
        nnet_copy.v_net.compile(
            optimizer = Adam(learning_rate = lr),
            loss = 'mean_squared_error'
        )
        return nnet_copy
    
    def save(self, name):
        self.v_net.save('models/' + name + '.h5')
