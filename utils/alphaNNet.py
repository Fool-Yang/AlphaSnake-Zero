from tensorflow import keras as ks


class AlphaNNet:
    
    def __init__(self, model = None, ins = None):
        if model:
            self.v_net = ks.models.load_model(model)
        elif ins:
            self.v_net = ks.Sequential([
                ks.layers.Conv2D(32, (5, 5), use_bias=False, input_shape=ins),
                ks.layers.BatchNormalization(axis=3),
                ks.layers.Activation('selu'),
                ks.layers.Conv2D(32, (3, 3), use_bias=False),
                ks.layers.BatchNormalization(axis=3),
                ks.layers.Activation('selu'),
                ks.layers.Conv2D(64, (3, 3), use_bias=False),
                ks.layers.BatchNormalization(axis=3),
                ks.layers.Activation('selu'),
                ks.layers.Conv2D(64, (3, 3), use_bias=False),
                ks.layers.BatchNormalization(axis=3),
                ks.layers.Activation('selu'),
                ks.layers.Conv2D(128,(3, 3), use_bias=False),
                ks.layers.BatchNormalization(axis=3),
                ks.layers.Activation('selu'),
                ks.layers.Conv2D(128,(3, 3), use_bias=False),
                ks.layers.BatchNormalization(axis=3),
                ks.layers.Activation('selu'),
                ks.layers.Flatten(),
                ks.layers.Dense(3, use_bias=False),
                ks.layers.BatchNormalization(),
                ks.layers.Activation('sigmoid')
            ])
    
    def train(self, X, Y, ep = None, bs = None):
        self.v_net.fit(X, Y, epochs = ep, batch_size = bs)
    
    def v(self, X):
        return self.v_net.predict(X)
    
    def copy(self, lr = 0.001):
        nnet_copy = AlphaNNet()
        # value
        nnet_copy.v_net = ks.models.clone_model(self.v_net)
        nnet_copy.v_net.build(self.v_net.layers[0].input_shape)
        nnet_copy.v_net.set_weights(self.v_net.get_weights())
        nnet_copy.v_net.compile(
            optimizer = ks.optimizers.Adam(learning_rate = lr),
            loss = "mean_squared_error"
        )
        return nnet_copy
    
    def save(self, name):
        self.v_net.save('models/' + name + '.h5')
    
    def remake(self, weights):
        new = ks.Sequential([
            ks.layers.Conv2D(32, (5, 5), use_bias=False, input_shape=self.v_net.layers[0].input_shape[1:]),
            ks.layers.BatchNormalization(axis=3),
            ks.layers.Activation('selu'),
            ks.layers.Conv2D(32, (3, 3), use_bias=False),
            ks.layers.BatchNormalization(axis=3),
            ks.layers.Activation('selu'),
            ks.layers.Conv2D(64, (3, 3), use_bias=False),
            ks.layers.BatchNormalization(axis=3),
            ks.layers.Activation('selu'),
            ks.layers.Conv2D(64, (3, 3), use_bias=False),
            ks.layers.BatchNormalization(axis=3),
            ks.layers.Activation('selu'),
            ks.layers.Conv2D(128,(3, 3), use_bias=False),
            ks.layers.BatchNormalization(axis=3),
            ks.layers.Activation('selu'),
            ks.layers.Conv2D(128,(3, 3), use_bias=False),
            ks.layers.BatchNormalization(axis=3),
            ks.layers.Activation('selu'),
            ks.layers.Flatten(),
            ks.layers.Dense(3),
            ks.layers.Activation('sigmoid')
        ])
        new.build(self.v_net.layers[0].input_shape)
        new.set_weights(weights)
        new.compile(
            optimizer = ks.optimizers.Adam(learning_rate = 0.001),
            loss = "mean_squared_error"
        )
        new.save('models/remake.h5')