import tensorflow as tf
from layers import *


class GDNN(object):
  def build_graph(self, placeholders, n=100, d=3, hidden_d=128,u=16, d_a=64, r=16,reuse=False):
    with tf.variable_scope('SelfAttentiveGraph', reuse=reuse):
      self.n = n
      self.d = d
      self.d_a = d_a
      self.u = u
      self.r = r
      self.dropout = placeholders['dropout']
      self.adj = placeholders['support']
      self.adj_inverse = placeholders['inverse_support']
      initializer = tf.keras.initializers.he_normal(seed = 123)
      self.input_F = placeholders['features']
      self.features_nonzero = placeholders['num_nodes']
      self.placeholders = placeholders
      
      hidden = GraphConvolutionSparse(input_dim=self.d,
                                      output_dim=hidden_d,
                                      adj=self.adj,
                                      features_nonzero=self.features_nonzero,
                                      act=tf.nn.relu,
                                      dropout=self.dropout,
                                      logging=False)(self.input_F)

      self.H = GraphConvolution(input_dim=hidden_d,
                                           output_dim=self.u,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=False)(hidden)
       
      self.W_s1 = tf.get_variable('W_s1', shape=[self.u,self.d_a],
          initializer=initializer)
      self.W_s2 = tf.get_variable('W_s2', shape=[self.d_a,self.r],
          initializer=initializer)

      self.batch_size = batch_size = tf.shape(self.input_F)[0]
      A = self.A = tf.nn.softmax(tf.matmul(tf.tanh(tf.matmul(self.H,self.W_s1)),self.W_s2))
      self.M = tf.matmul(tf.transpose(A), self.H)
      A_T = tf.transpose(A, perm=[1, 0])
      AA_T = tf.matmul(A_T,A) - tf.eye(r)
      self.P = tf.square(tf.norm(AA_T))

      inverse_H = tf.matmul(A,self.M)
      de_hidden = GraphConvolution(input_dim=self.u,
                                           output_dim=hidden_d,
                                           adj=self.adj_inverse,
                                           act=tf.nn.relu,
                                           dropout=self.dropout,
                                           logging=False)(inverse_H)
      self.reconstruct_X = GraphConvolution(input_dim=hidden_d,
                                           output_dim=self.d,
                                           adj=self.adj,
                                           act=lambda x:x,
                                           dropout=self.dropout,
                                           logging=False)(de_hidden)



  def trainable_vars(self):
    return [var for var in
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='SelfAttentiveGraph')]
