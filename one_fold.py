import tensorflow as tf
import numpy as np
from utils import *
from gdnn import GDNN
import networkx as nx
from load_data import load_nci
from evaluate_embedding import evaluate_embedding
# Newloader
from graph.dataset import load




def fold_cv(data_index, FLAGS,seed=123,epoch_n=20):
  seed = seed
  if_train = FLAGS.train
  FLAGS = tf.app.flags.FLAGS
  cv_index = FLAGS.cv_index
  num_epochs = epoch_n
  tag_size = FLAGS.labels
  graph_pad_length = FLAGS.graph_pad_length
  feature_dimension = FLAGS.feature_dimension
  CE_ratio = FLAGS.CE_ratio
  lr = FLAGS.learn_rate

  placeholders = {
    'support': tf.sparse_placeholder(tf.float32),
    'inverse_support': tf.sparse_placeholder(tf.float32),
    'features':tf.sparse_placeholder(tf.float32),
    'labels': tf.placeholder(tf.float32, shape=(graph_pad_length,feature_dimension)),
    'num_nodes': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'g_labels': tf.placeholder(tf.float32, shape=())
  }
  # construct graph for graph classification
  with tf.Session() as sess:
    model = GDNN()
    model.build_graph(n=graph_pad_length,placeholders = placeholders,d =feature_dimension,seed=seed)
    with tf.variable_scope('DownstreamApplication'):
      global_step = tf.Variable(0, trainable=False, name='global_step')
      learn_rate = tf.train.exponential_decay(lr, global_step, FLAGS.decay_step, 0.98, staircase=True)
      layer_flat = tf.reshape(model.M,[1,-1])
      labels = placeholders['labels'][:placeholders['num_nodes']] 
      logits = model.reconstruct_X[:placeholders['num_nodes']]
      loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=labels,logits=logits,pos_weight = CE_ratio*feature_dimension)) 
      p_coef = 0.0
      p_loss = p_coef * model.P
      loss = loss + p_loss
      params = tf.trainable_variables()
  
    # load data
    if data_index == "proteins" or data_index =="DD":
      raw_train_structure_input,raw_train_feature_input,raw_test_structure_input,raw_test_feature_input,ally,ty = load_nci(data_index)
      # graph padding
      train_structure_input,train_feature_input = graph_padding(raw_train_structure_input,raw_train_feature_input,graph_pad_length,feature_dimension=feature_dimension)
      test_structure_input,test_feature_input = graph_padding(raw_test_structure_input,raw_test_feature_input,graph_pad_length,feature_dimension = feature_dimension)
    else:
      train_structure_input, diff, train_feature_input, ally, num_nodes_all = load(data_index)
      test_structure_input, diff, test_feature_input, labels, num_nodes = load(data_index)
    total = len(train_feature_input)
  
    optimizer = tf.train.AdamOptimizer(learn_rate)
    grad_and_vars = tf.gradients(loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(grad_and_vars, 0.5)
    opt = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)
    sess.run(tf.global_variables_initializer())
    train_emb = []
    if if_train == True:
      hist_loss = []
      for epoch_num in range(num_epochs):
        epoch_loss = 0
        step_loss = 0
        for i in range(int(total)):
          if data_index == "proteins" or data_index == "DD":
            num_nodes = raw_train_feature_input[i].shape[0]
          else:
            num_nodes = num_nodes_all[i]
          batch_input,topo, batch_tags,g_l = (train_feature_input[i],train_structure_input[i], train_feature_input[i].todense(),ally[i])
          batch_input = preprocess_features(batch_input.tolil())
          batch_topo = preprocess_adj(topo)
          batch_topo_inverse = preprocess_inverse_adj(topo)
          train_ops = [opt, loss, learn_rate, global_step]
          train_ops += [p_loss]
          feed_dict = construct_feed_dict(batch_input, batch_topo, batch_topo_inverse,batch_tags, num_nodes,g_l,placeholders)
          result = sess.run(train_ops, feed_dict=feed_dict)
          step_loss += result[1]
          epoch_loss += result[1]
          step_loss = 0
        print("Epoch:", '%04d' % (epoch_num), "train_loss=", "{:.5f}".format(epoch_loss))
        if epoch_num == num_epochs -1:
          for i in range(int(total)):
            if data_index == "proteins" or data_index == "DD":
              num_nodes = raw_train_feature_input[i].shape[0]
            else:
              num_nodes = num_nodes_all[i]
            batch_input,topo, batch_tags,g_l = (train_feature_input[i],train_structure_input[i], train_feature_input[i].todense(),ally[i])
            batch_input = preprocess_features(batch_input.tolil())
            batch_topo = preprocess_adj(topo)
            batch_topo_inverse = preprocess_inverse_adj(topo)
            train_ops = [opt, loss, learn_rate, global_step]
            train_ops += [p_loss]
            feed_dict = construct_feed_dict(batch_input, batch_topo, batch_topo_inverse,batch_tags, num_nodes,g_l,placeholders)
            result = sess.run(tf.reshape(layer_flat,[-1]), feed_dict=feed_dict)
            train_emb.append(result)
          train_emb = np.array(train_emb)
          np.save('embeddings',train_emb)
      sess.close()
    else:
      pass

    if if_train == True:
      prediction = evaluate_embedding(train_emb,ally)
      return prediction
    else:
      pass



