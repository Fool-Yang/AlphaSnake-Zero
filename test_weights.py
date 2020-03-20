from utils.alphaNNet import AlphaNNet
from numpy import max, min, power, sum

file_name = input("\nEnter the model name:\n")
net = AlphaNNet(model = "models/" + file_name + ".h5")

W = net.v_net.get_weights()
for w in W:
    print(w.shape)
    print(min(w), max(w))
    print(sum(power(w, 2)))
    print()