from utils.alpha_nnet import AlphaNNet
from numpy import max, min, power, sum

file_name = input("\nEnter the model name:\n")
net = AlphaNNet(model_name = "models/" + file_name + ".h5")

W = net.v_net.get_weights()
for w in W:
    print(w.shape)
    print("Min weight:", min(w), "Max weight:", max(w))
    print("Sum of squres (L2)", sum(power(w, 2)))
    print()
