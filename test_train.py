from utils.alpha_snake_zero_trainer import AlphaSnakeZeroTrainer
from utils.alphaNNet import AlphaNNet
start = int(input("Enter starting iteration:\n"))
if start == 0:
    net = AlphaNNet(input_shape = (21, 21, 3))
else:
    net = AlphaNNet(model = "models/nn" + str(start) + ".h5")
a = AlphaSnakeZeroTrainer()
a.train(net, iter = start)