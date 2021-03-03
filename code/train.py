from utils.alpha_snake_zero_trainer import AlphaSnakeZeroTrainer
from utils.alphaNNet import AlphaNNet
import tensorflow as tf

try:
    # when running on Google Cloud
    # the TPU must be located in the same area as the CPU
    # pass the TPU's name
    Resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu = "node")
    tf.config.experimental_connect_to_cluster(Resolver)
    tf.tpu.experimental.initialize_tpu_system(Resolver)
    TPU = tf.distribute.experimental.TPUStrategy(Resolver)
except:
    print("Cannot find the Google Cloud TPU. Using CPUs")
    TPU = None

name = input("Enter the model name (not including the generation number):\n")
start = int(input("Enter the starting iteration (0 for creating a new model):\n"))
if start == 0:
    ANNet = AlphaNNet(input_shape = (21, 21, 3))
else:
    ANNet = AlphaNNet(model_name = "models/" + name + str(start) + ".h5")
Trainer = AlphaSnakeZeroTrainer(TPU = TPU)
Trainer.train(ANNet, name = name, iteration = start)
