import os
import sys
import tensorflow as tf
import fadernet_tf

fadernet = fadernet_tf.Fadernet(batch_size=32)

fadernet.train()

fadernet.test()