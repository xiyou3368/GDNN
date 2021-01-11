import numpy as np
from one_fold import fold_cv
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_epochs', 20, 'number of epochs to train')
flags.DEFINE_integer('labels', 1, 'number of label classes')
flags.DEFINE_integer('graph_pad_length', 64, 'graph pad length for training')
flags.DEFINE_integer('feature_dimension', 64, 'feature dimension of the graph')
flags.DEFINE_float('CE_ratio', 1.0, 'balancing ratio of the CE loss')
flags.DEFINE_integer('decay_step', 100, 'decay steps')
flags.DEFINE_integer('cv_index', 2, 'fold_ID')
flags.DEFINE_float('learn_rate', 1e-2, 'learn rate for training optimization')
flags.DEFINE_boolean('train',False, 'train mode FLAG')

configs = {
  'IMDB-BINARY': {
                'feature_dimension': 136,
                'graph_pad_length': 136,
                'CE_ratio': 1.0,
               }
}



def read_config(config, FLAGS):
  for item in config.keys():
    if item in FLAGS.__flags:
      FLAGS.__flags[item].value = config[item]



temp = []
dataset = ["IMDB-BINARY"]
for i in dataset:
  config = configs[i]
  read_config(config, FLAGS)
  tf.reset_default_graph()
  seed = 4
  np.random.seed(seed)
  tf.set_random_seed(seed)
  
  ttemp = fold_cv(i, FLAGS)
  temp.append(ttemp)
  print(i,ttemp)
