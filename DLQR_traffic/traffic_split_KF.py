import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import time
import os
import matplotlib.pyplot as plt

class KalmanFilter(object):
    """
    This class defines a kalman filter

    l - latent state
    l_a_priori - A priori state estimate
    l_a_posteriori - A posteriori state estimate

    P_a_priori - A priori error covariance
    P_a_posteriori - A posteriori error covariance

    F - state-transition model
    Q - covariance of the process noise    
    a, b - observation model and bias
    R - covariance of the observation noise
    z - observation

    y_pre - measurement pre-fit residual
    S - Pre-fit residual covariance
    K - Kalman gain
    y_post - measurement post-fit residual
    """
    
    def __init__(self, dim_l, dim_z, batch_size, **kwargs):
        self.dim_l = dim_l
        self.dim_z = dim_z
        self.dim_y = dim_z

        # lambda initializer for identity matrices
        self.eye_init = lambda shape, dtype = np.float32: np.eye(*shape, dtype = dtype)

        self._I = tf.constant(self.eye_init((dim_l, dim_l)), name= 'I')


        '''This section requires these kwargs to exist, cannot handle missing args'''


        '''This section also cannot handle missing kwargs'''
        self.l_0 = kwargs.pop('l_0', None)
        self.P_0 = kwargs.pop('P_0', None)
        self.F = kwargs.pop('F', None)
        self.g = kwargs.pop('g', None)
        self.a = kwargs.pop('a', None)
        self.b = kwargs.pop('b', None)
        self.sigma = kwargs.pop('sigma', None)
        self.y_0 = kwargs.pop('y_0', None)
        self.z_0 = kwargs.pop('z_0', None)
        self.pred_0 = kwargs.pop('pred_0', None)
        self.z = kwargs.pop('z', None)
        self.g_pred = kwargs.pop('g_pred', None)
        self.sigma_pred = kwargs.pop('sigma_pred', None)
        self.l_0_pred = kwargs.pop('l_0_pred', None)
        self.z_0_pred = kwargs.pop('z_0_pred', None)
        self.F_pred = kwargs.pop('F_pred', None)
        self.a_pred = kwargs.pop('a_pred', None)
        self.b_pred = kwargs.pop('b_pred', None)


    def forward_filter_fn(self, params, inputs):
        """
        Forward step over a batch
        params contains: l_a_posteriori, P_a_posteriori, y_pre
        inputs contains: z, F, g, sigma, a, b

        Calculates prior distributions based on the given posterior distributions and the current residual
                updates posterior distributions based on the new prior distributions
        """
        '''Shapes:
            z = (bs, dim_z)
            l_a_posteriori = (bs, dim_l, dim_z)
            P_a_posteriori = (bs, dim_l, dim_l)
            F = (bs, dim_l, dim_l)
            Q = (bs, dim_l, dim_l)
            R = (bs, dim_z, dim_z)
            a = (bs, dim_l, dim_z)
            b = (bs, dim_z)
        '''
        
        z, F, g, sigma, a, b = inputs
        l_a_posteriori, P_a_posteriori, y_pre, pred = params


        
        l_a_priori = tf.matmul(F,l_a_posteriori)
        P_a_priori = tf.matmul(tf.matmul(F,P_a_posteriori), F, transpose_b = True) + tf.matmul(g,g, transpose_b=True)


        y_pre = tf.expand_dims(z - tf.squeeze(tf.add(tf.matmul(a, l_a_priori, transpose_a=True), b),-1),-1)

        S = tf.matmul(sigma, sigma, transpose_b=True) + \
            tf.matmul(tf.matmul(a, P_a_priori, transpose_a=True), a)
        S_inv = tf.reciprocal(S)
        '''TODO: Compute inverse using cholesky decomposition? Only works if a is matrix
                so z must be multivariate
        '''
        
        K = tf.matmul(tf.matmul(P_a_priori, a), S_inv)
        l_a_posteriori = l_a_priori + tf.matmul(K,y_pre)
        I_Ka = self._I-tf.matmul(K,a, transpose_b=True)
        P_a_posteriori = tf.matmul(tf.matmul(I_Ka, P_a_priori), I_Ka, transpose_b=True) + \
                         tf.matmul(tf.matmul(K,tf.matmul(sigma, sigma, transpose_b = True)),
                                   K, transpose_b=True)
        y_post = z - tf.squeeze(tf.add(tf.matmul(a, l_a_posteriori, transpose_a=True), b), -1)
        pred = tf.squeeze(tf.add(tf.matmul(a, l_a_posteriori, transpose_a=True),b), -1)
        return l_a_posteriori, P_a_posteriori, y_post, pred

    def forward_filter(self):
        """
        Compute the forward step in Kalman Filter
        The forward pass is initialized with p(x_1) = N(self.x, self.P)
        We return the mean and covariance for p(x_t|x_tm1) for t=2, ..., T+1
        and the filtering distribution p(x_t|z_1:t) for t=1, ..., T
        """

        forward_states = tf.scan(self.forward_filter_fn,
                                 elems = (trans(self.z),trans(self.F),
                                          trans(self.g),trans(self.sigma),
                                          trans(self.a),trans(self.b)),
                                initializer=(self.l_0, self.P_0, self.y_0, self.pred_0))
        
        return forward_states
    
    def Kfilter(self):
        l_filtered, P_filtered, residuals, filtered_prediction = self.forward_filter()
        return trans(l_filtered), trans(P_filtered), trans(residuals), trans(filtered_prediction)
        
    def forward_predict_fn(self, params, inputs):
        """Forward step over a batch
        params contains l_prev, z_prev
        inputs contains F, g, a, b, sigma"""
        
        F, g, a, b, sigma = inputs
        l_prev, z_prev = params
        
        l_next = tfd.MultivariateNormalDiag(loc = tf.matmul(F, l_prev), scale_diag = g).sample()
        z_next = tfd.Normal(loc = tf.matmul(a, l_prev, transpose_a=True)+b, scale = sigma).sample()
        return l_next, z_next
    
    def forward_predict(self):
        """
        Compute the predictions in state space model
        The forward pass is initialized by l_T = p(l_T|z_1:T)
        We return the hidden states l_T+1:T+t and predictions z_T+1:T+t
        """
        
        forward_predictions = tf.scan(self.forward_predict_fn,
                                      elems = (trans(self.F_pred), trans(self.g_pred),
                                               trans(self.a_pred), trans(self.b_pred),
                                               trans(self.sigma_pred)),
                                      initializer = (self.l_0_pred, self.z_0_pred))
        
        return forward_predictions
        
    def Kpredict(self):
        
        l_predicted, z_predicted = self.forward_predict()
        return trans(l_predicted), trans(z_predicted)
    
def trans(tensor):
    if len(tensor.shape)==3:
        return tf.transpose(tensor, [1,0,2])
    else:
        return tf.transpose(tensor, [1,0,2,3])

class NameError(Exception):
    pass

def find_avg(parameter):
    '''Must be passed a numpy array'''
    return np.mean(np.reshape(parameter, [-1]+[x for x in parameter.shape[2:]]), axis = 0)

class LSTM_SSM_model(object):
    def __init__(self, sess, name, dim_l = 31, dim_z = 1, num_samples = 963, feature_len = 994,
                 learning_rate = 0.00001, lr_decay = 0.95, sigma_upper_bound = 0.1,
                 sigma_lower_bound = 0.001, g_upper_bound = 0.1,
                 g_lower_bound = 0.01, mu_0_upper_bound = 0.9,mu_0_lower_bound = 0,
                 Sigma_0_upper_bound = 0.5, Sigma_0_lower_bound = 0, beta = 0.00001,
                 b_upper_bound = 0.05, b_lower_bound = -0.05
                ):
        if name == '':
            raise NameError("A model has no name")

            
        self.sess = sess
        
        self.learning_rate = learning_rate
        self.batch_size = 32
        self.train_range = 672
        self.test_range = 168
        self.num_samples, self.sample_len, feature_len = (num_samples,self.train_range,feature_len)
        self.num_batches = self.num_samples//self.batch_size
        self.global_step = tf.Variable(0, name='global_step',trainable=False)
        self.increment_global_step = tf.assign_add(self.global_step,1, name = 'increment_global_step')
        self.lr_decay = lr_decay
#        self.decayed_learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
#                                                               self.num_batches, self.lr_decay)

        
        self.dim_l = dim_l # seasonal model, 7 days
        self.dim_z = dim_z
        self.initial_variance = 1
        
        self.sigma_upper_bound = sigma_upper_bound
        self.g_lower_bound = g_lower_bound
        self.g_upper_bound = g_upper_bound
        self.mu_0_upper_bound = mu_0_upper_bound
        self.mu_0_lower_bound = mu_0_lower_bound
        self.Sigma_0_upper_bound = Sigma_0_upper_bound
        self.Sigma_0_lower_bound= Sigma_0_lower_bound
        self.b_upper_bound = b_upper_bound
        self.b_lower_bound = b_lower_bound
        self.beta = beta
        
        self.lstm_sizes = [128,64]
        last_lstm = self.lstm_sizes[-1]
        
        self.model_folder = 'traffic/{}'.format(name)
        if not os.path.isdir(self.model_folder):
            print('This model has no folder')
            os.makedirs(self.model_folder)
        self.saved_model_location = '{}/model.ckpt'.format(self.model_folder)
#        self.log_dir = 'log/'+name+time.ctime().replace(' ','_')

        self.losses = []
        self.saver = None
        
        with tf.variable_scope('KF', reuse = tf.AUTO_REUSE):
#             self.F = tf.get_variable(initializer = tf.tile(tf.expand_dims(tf.expand_dims(tf.eye(self.dim_l),0),0),
#                                              (self.batch_size,self.sample_len,1,1)),dtype = tf.float32, name = 'F', trainable = False)
            self.W_F = tf.get_variable(initializer = tf.random.normal([self.batch_size,self.dim_l*self.dim_l,last_lstm]),
                                       dtype = tf.float32, name = 'W_F')
            self.bias_F = tf.get_variable(initializer = tf.zeros([self.batch_size, self.dim_l, self.dim_l]),
                                         dtype = tf.float32, name = 'bias_F')
            self.W_a = tf.get_variable(initializer = tf.random.normal([self.batch_size, self.dim_l, last_lstm]),
                                       dtype = tf.float32, name = 'W_a')
            self.bias_a = tf.get_variable(initializer = tf.zeros([self.batch_size, self.dim_l, self.dim_z]),
                                          dtype = tf.float32, name = 'bias_a')
            
            self.W_b = tf.get_variable(initializer = tf.random.normal([self.batch_size, self.dim_z, last_lstm]),
                                       dtype = tf.float32, name = 'W_b')
            self.bias_b = tf.get_variable(initializer = tf.zeros([self.batch_size, self.dim_z, self.dim_z]),
                                         dtype = tf.float32, name = 'bias_b')

            self.W_g = tf.get_variable(initializer = tf.random.normal([self.batch_size, self.dim_l,
                                                                       last_lstm]),
                                       dtype = tf.float32, name = 'W_g')
            self.bias_g = tf.get_variable(initializer = tf.zeros([self.batch_size, self.dim_l, 1]),
                                          dtype = tf.float32, name = 'bias_g')

            self.W_sigma = tf.get_variable(initializer = tf.random.normal([self.batch_size, self.dim_z,
                                                                           last_lstm]),
                                           dtype = tf.float32, name = 'W_sigma')
            self.bias_sigma = tf.get_variable(initializer = tf.zeros([self.batch_size, self.dim_z, 1]),
                                              dtype = tf.float32, name = 'bias_sigma')

            self.W_mu_0 = tf.get_variable(initializer = tf.random.normal([self.batch_size, self.dim_l,
                                                                          last_lstm]),
                                         dtype = tf.float32, name = 'W_mu_0')
            self.bias_mu_0 = tf.get_variable(initializer = tf.zeros([self.batch_size, self.dim_l, 1]),
                                            dtype = tf.float32, name = 'bias_mu_0')
            
            self.W_Sigma_0 = tf.get_variable(initializer = tf.random.normal([self.batch_size, self.dim_l,
                                                                             last_lstm]),
                                            dtype = tf.float32, name = 'W_Sigma_0')
            self.bias_Sigma_0 = tf.get_variable(initializer = tf.zeros([self.batch_size, self.dim_l, 1]),
                                                dtype = tf.float32, name = 'bias_Sigma_0')
            
            
            self.P_0 = tf.Variable(tf.tile(tf.expand_dims(self.initial_variance*tf.eye(self.dim_l,dtype = tf.float32),0),
                               (self.batch_size, 1, 1)), name = 'P_0', trainable = False)

            self.y_0 = tf.Variable(tf.zeros([self.batch_size, self.dim_z]), dtype = tf.float32, name = 'y_0', trainable = False)
            self.z_0 = tf.Variable(tf.zeros([self.batch_size, self.dim_z, self.dim_z]), dtype = tf.float32, name = 'z_0', trainable = False)
            self.pred_0 = tf.Variable(tf.zeros([self.batch_size, self.dim_z]), dtype = tf.float32, name = 'pred_0', trainable = False)
            
            self.F_test = tf.placeholder(tf.float32, shape = [self.batch_size,self.test_range, self.dim_l, self.dim_l], name = 'F_test')
            self.a_test = tf.placeholder(tf.float32, shape = [self.batch_size, self.test_range, self.dim_l, self.dim_z], name = 'a_test')
            self.b_test = tf.placeholder(tf.float32, shape = [self.batch_size, self.test_range, self.dim_z, 1], name = 'b_test')
            self.g_test = tf.placeholder(tf.float32, shape = [self.batch_size, self.test_range, self.dim_l, 1], name = 'g_test')
            self.sigma_test = tf.placeholder(tf.float32, shape = [self.batch_size, self.test_range, self.dim_z, 1], name = 'sigma_test')
            self.l_0_test = tf.placeholder(tf.float32, shape = [self.batch_size, self.dim_l, 1], name = 'l_0_test')
            self.final_z = tf.placeholder(tf.float32, shape = [self.batch_size, 1, self.dim_z], name = 'final_z')
            
        with tf.variable_scope('LSTM', reuse = tf.AUTO_REUSE):
            self.lstm_input = tf.placeholder(tf.float32, shape= [None, self.sample_len, feature_len], name = 'lstm_input')

        self.z = tf.placeholder(tf.float32, shape = [None, self.sample_len, self.dim_z], name = 'z')
        
    def build_LSTM(self, prev_state = None):
        with tf.name_scope('LSTM'):
            with tf.variable_scope('LSTM', reuse=tf.AUTO_REUSE):

                lstms = [tf.contrib.rnn.LSTMCell(size, reuse=tf.get_variable_scope().reuse) for size in self.lstm_sizes]
                dropouts = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob = 0.5) for lstm in lstms]

                cell = tf.contrib.rnn.MultiRNNCell(dropouts)
                if prev_state:
                    initial_state = prev_state
                else:
                    initial_state = cell.zero_state(self.batch_size, tf.float32)
                self.lstm_output, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_input, initial_state = initial_state)
        return self

    def affine_transformations(self):

        with tf.variable_scope('affine_transformations'):
            
            self.lstm_output = tf.expand_dims(self.lstm_output, -1) # (32, 672, 64, 1)

            def tile_func(Weights_or_bias):
                if len(Weights_or_bias.shape) == 2:
                    return tf.tile(tf.expand_dims(Weights_or_bias,1), (1,self.sample_len,1))
                elif len(Weights_or_bias.shape) == 3:
                    return tf.tile(tf.expand_dims(Weights_or_bias,1), (1,self.sample_len,1,1))
                else:
                    raise ValueError('Unknown sized Weights or bias array')
                    
            '''TODO: Does tile work by updating all weights for the tile back to the original matrix?    
            '''
            
            W_F = tile_func(self.W_F)
            bias_F = tile_func(self.bias_F)
            W_a = tile_func(self.W_a)
            bias_a = tile_func(self.bias_a)
            W_b = tile_func(self.W_b)
            bias_b = tile_func(self.bias_b)
            W_g = tile_func(self.W_g)
            bias_g = tile_func(self.bias_g)
            W_sigma = tile_func(self.W_sigma)
            bias_sigma = tile_func(self.bias_sigma)

    
            self.F = tf.add(tf.reshape(tf.matmul(W_F,self.lstm_output),[self.batch_size,self.sample_len,self.dim_l,self.dim_l]),bias_F)
    
            self.a = tf.add(tf.matmul(W_a, self.lstm_output), bias_a)

            temp_b = tf.add(tf.matmul(W_b, self.lstm_output), bias_b)
            self.b = ((self.b_upper_bound-self.b_lower_bound)/(1+tf.exp(-temp_b)))+self.b_lower_bound

            transition_error = tf.add(tf.matmul(W_g, self.lstm_output), bias_g)
            self.g = ((self.g_upper_bound-self.g_lower_bound)/(1+tf.exp(-transition_error)))+self.g_lower_bound
    
            observation_error = tf.add(tf.matmul(W_sigma, self.lstm_output), bias_sigma)
            self.sigma = (self.sigma_upper_bound)/(1+tf.exp(-observation_error))
            
            temp_mu_0 = tf.add(tf.matmul(self.W_mu_0, self.lstm_output[:,-1,:]), self.bias_mu_0)
            self.mu_0 = ((self.mu_0_upper_bound-self.mu_0_lower_bound)/(1+tf.exp(-temp_mu_0)))+self.mu_0_lower_bound

            temp_Sigma_0 = tf.add(tf.matmul(self.W_Sigma_0, self.lstm_output[:,-1,:]), self.bias_Sigma_0)
            self.Sigma_0 = ((self.Sigma_0_upper_bound-self.Sigma_0_lower_bound)/(1+tf.exp(-temp_Sigma_0)))+self.Sigma_0_lower_bound
            
            l_0_distribution = tfd.MultivariateNormalDiag(loc = self.mu_0, scale_diag = self.Sigma_0)

            self.l_0 = l_0_distribution.sample()
        return self
    
    def build_model(self):
        self.kf_train = KalmanFilter(batch_size=self.batch_size,
                                     dim_l=self.dim_l,
                                     dim_z=self.dim_z,
                                     l_0 = self.l_0,
                                     P_0 = self.P_0,
                                     F = self.F,
                                     g = self.g,
                                     a = self.a,
                                     b = self.b,
                                     sigma = self.sigma,
                                     z = self.z,
                                     y_0 = self.y_0,
                                     pred_0 = self.pred_0
                                    )
        with tf.variable_scope('KF_results', reuse=tf.AUTO_REUSE):
            self.l_filtered, self.P_filtered, self.residuals, self.filtered_predictions = self.kf_train.Kfilter()

        return self
    
    def likelihood_fn(self, params, inputs):
        '''Compute likelihood over a batch
        params contains: mu, Sigma - the parameters of the likelihood distribution
        inputs contains: calculations of mu: F, a, f==l_a_posteriori  <-???
                        calculations of Sigma: a, F, S == P_a_posteriori????, g, sigma
        '''
        a, b, F, g, sigma, f, S = inputs
        mu, Sigma = params
        '''
        a (bs, dim_l, 1)
        b (bs, 1)
        F (bs, dim_l, dim_l)
        g (bs, dim_l, 1)
        sigma (bs, 1, 1)
        f (bs, dim_l, 1)
        S (bs, dim_l, dim_l)
        mu (bs, 1, 1)
        Sigma (bs, 1, 1)
        '''

        mu = tf.add(tf.matmul(tf.matmul(a, F, transpose_a=True), f), b)

        temp = tf.matmul(tf.matmul(F, S), F, transpose_b=True) + tf.matmul(g, g, transpose_b=True)
        Sigma = tf.matmul(tf.matmul(a, temp, transpose_a=True), a) + tf.square(sigma)
        
        return mu, Sigma
    
    def build_loss(self):
        '''Useful shapes(Ideally):
            l_a_posteriori(batch) - (batch_size, sample_len, dim_l)
            P_a_posteriori(batch) - (batch_size, sample_len, dim_l,dim_l)
            
            inputs:
                mu_0, a, F, l_a_posteriori?
                Sigma_0, a, R, F, P_a_posteriori, Q
        '''


        with tf.variable_scope('loss', reuse = tf.AUTO_REUSE):
            decayed_learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                       self.num_batches, self.lr_decay)


                
            mu_1 = tf.add(tf.matmul(trans(self.a)[0], self.mu_0, transpose_a=True),trans(self.b)[0])
            Sigma_1 = tf.matmul(tf.matmul(trans(self.a)[0], tf.linalg.diag(tf.squeeze(self.Sigma_0)),
                                          transpose_a=True),
                                trans(self.a)[0]) + tf.square(trans(self.sigma)[0])
    
            mu, Sigma = tf.scan(self.likelihood_fn,
                                elems = (trans(self.a)[1:], trans(self.b)[1:],
                                         trans(self.F)[1:], trans(self.g)[1:],
                                         trans(self.sigma)[1:],trans(self.l_filtered)[:-1],
                                         trans(self.P_filtered)[1:]),
                                initializer = (mu_1, Sigma_1))
            self.mu = tf.concat([mu_1, tf.squeeze(trans(mu),-1)], 1)
            self.Sigma = tf.concat([Sigma_1, tf.squeeze(trans(Sigma),-1)], 1)

            z_distribution = tfd.Normal(loc = self.mu, scale = self.Sigma)
            self.z_probability = z_distribution.prob(self.z)

            regularizers = tf.nn.l2_loss(self.W_g) + tf.nn.l2_loss(self.W_mu_0) + \
                        tf.nn.l2_loss(self.W_sigma) + tf.nn.l2_loss(self.W_Sigma_0) + \
                        tf.nn.l2_loss(self.W_a) + tf.nn.l2_loss(self.W_b)
            self.loss = tf.reduce_mean(self.beta*regularizers)-tf.reduce_sum(tf.log(self.z_probability+1e-8))
            tf.summary.scalar('loss', self.loss)

            self.optimizer = tf.train.AdamOptimizer(decayed_learning_rate)
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            capped_grads_and_vars = [(tf.clip_by_norm(grad, 1.), var) for grad, var in grads_and_vars]
            self.train_op = self.optimizer.apply_gradients(capped_grads_and_vars)
        return self
    
    def initialize_variables(self):
        self.saver = tf.train.Saver()
        try:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_folder))
            print("Restoring model from {}".format(self.saved_model_location))
        except:
            self.sess.run(tf.global_variables_initializer())
            print("Initializing new model at {}".format(self.saved_model_location))
        return self

    
    def train(self, epochs):
        start = time.time()
#        merged = tf.summary.merge_all()
#        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        for i in range(epochs):
            preds = []
            epoch_loss = []
            perm = np.random.permutation(self.num_samples)
            for idx in range(self.num_batches):


                slc = np.array([perm[j] for j in range(idx*self.batch_size, (idx+1)*self.batch_size)])
                lane_id_onehot = np.zeros([self.batch_size, self.train_range, self.num_samples])
                for k in range(self.batch_size):
                    lane_id_onehot[k, :,slc[k]] = 1
                train_x = np.concatenate([self.batch_x_seasonality, lane_id_onehot], axis = 2)

                feed_dict = {self.lstm_input: train_x,self.z: self.train_z[slc]}
#                summary, loss_, _, _ = self.sess.run([merged, self.loss, self.train_op, self.increment_global_step],
#                                                  feed_dict=feed_dict)
                loss_, _, _ = self.sess.run([self.loss, self.train_op, self.increment_global_step],
                                            feed_dict=feed_dict)
                epoch_loss.append(loss_)
                print(tf.trainable_variables())
#                writer.add_summary(summary, self.sess.run(self.global_step))
#                if idx % 10 == 0:
#                  print('mini epoch #{}'.format(idx))

            epoch_loss = np.mean(epoch_loss)
            self.losses.append(epoch_loss)
            print("Epoch #{}\tTime Elapsed: {}\tNegative Log-Likelihood {}".
                  format(self.sess.run(self.global_step)/self.num_batches,
                         (time.time()-start)/60, epoch_loss))
            if (i+1)%50 == 0:
                self.saver.save(self.sess, self.saved_model_location, global_step = self.global_step)
                print("Model Saved at {}".format(self.saved_model_location))
#        self.saver.save(self.sess, self.saved_model_location, global_step = self.global_step)
#        print("Model Saved at {}".format(self.saved_model_location))
        return self.losses
    
    def get_test_variables(self):
        start = time.time()
        preds = []
        gs = []
        sigmas = []
        Q50_numerator = 0
        Q90_numerator = 0
        
        for idx in range((self.num_samples//self.batch_size)):
            beg,end = [idx*self.batch_size, (idx+1)*self.batch_size]
            
            # Get the final hidden state from kalman filter
            lane_id_onehot = np.zeros([self.batch_size, self.train_range, self.num_samples])
            for k in range(beg, end):
                lane_id_onehot[k%self.batch_size, :, k] = 1
            train_x = np.concatenate([self.batch_x_seasonality, lane_id_onehot], axis = 2)
            
            feed_dict = {self.lstm_input: train_x, self.z: self.train_z[beg:end]}
        
            filtered_train, g_train, sigma_train, self.l_filtered_test, self.final_lstm_state = self.sess.run([self.filtered_predictions, self.g, self.sigma,
                                                                                                               self.l_filtered, self.final_state],
                                                                                                              feed_dict=feed_dict)

            l_0_test = self.l_filtered_test[:,-1,:,:]
            
            self.build_LSTM(self.final_lstm_state).affine_transformations()
            
            feed_dict = {self.lstm_input: train_x}
            g_test, sigma_test, a_test, b_test, F_test= self.sess.run([self.g, self.sigma, self.a, self.b, self.F],
                                                         feed_dict=feed_dict)
            feed_dict = {self.final_z: self.train_z[beg:end,-1,:,np.newaxis], self.l_0_test: l_0_test,
                         self.g_test: g_test[:,:self.test_range], self.sigma_test: sigma_test[:,:self.test_range],
                         self.a_test: a_test[:,:self.test_range], self.b_test: b_test[:,:self.test_range],
                         self.F_test: F_test[:,:self.test_range]
                        }
            
            z_preds = np.squeeze(self.sess.run([self.z_preds_test], feed_dict=feed_dict)[0])
            preds.append(np.concatenate([np.squeeze(filtered_train),z_preds], axis = 1))
            gs.append(np.squeeze(np.concatenate([g_train,g_test[:,:self.test_range]], axis = 1)))
            sigmas.append(np.squeeze(np.concatenate([sigma_train,sigma_test[:,:self.test_range]], axis = 1)))
            
            for i in range(self.batch_size):
                for j in range(len(z_preds[i])):
                    Q50_numerator += self.P(0.5, z_preds[i][j], self.test_z[beg+i][j])
                    Q90_numerator += self.P(0.9, z_preds[i][j], self.test_z[beg+i][j])

        preds = np.asarray(preds)
        preds = np.reshape(preds, (-1,preds.shape[-1]))
        gs = np.asarray(gs)
        gs = np.reshape(gs, (-1, gs.shape[-2], gs.shape[-1]))
        sigmas = np.asarray(sigmas)
        sigmas = np.reshape(sigmas, (-1, sigmas.shape[-1]))
        Q_denomenator = np.sum(np.abs(self.test_z))
        pq50_loss = 2*Q50_numerator/Q_denomenator
        pq90_loss = 2*Q90_numerator/Q_denomenator
        
        return preds, gs, sigmas, pq50_loss, pq90_loss
    
    
    def build_test(self):
        self.kf_test = KalmanFilter(batch_size=self.batch_size,
                                    test_range = self.test_range,
                                    dim_l=self.dim_l,
                                    dim_z=self.dim_z,
                                    l_0_pred = self.l_0_test,
                                    g_pred = self.g_test,
                                    sigma_pred = self.sigma_test,
                                    F_pred = self.F_test,#F_pred = self.F[:,:self.test_range],
                                    a_pred = self.a_test,
                                    b_pred = self.b_test,
                                    z_0_pred = self.final_z
                                   )
        with tf.name_scope('KF_Predictions'):
            with tf.variable_scope('KF_Predictions'):
                self.l_preds_test, self.z_preds_test = self.kf_test.Kpredict()
        return self
            
    def load_data(self):
        all_z = np.load('formatted_traffic.npy')
        train_range = 28
        self.train_z = np.reshape(all_z[:,:train_range,:], [all_z.shape[0], -1, 1])
        self.test_z = np.reshape(all_z[:,train_range:,:], [all_z.shape[0], -1])
        del all_z
        x_seasonality = np.load('x_seasonality.npy')
        self.batch_x_seasonality = np.repeat(np.expand_dims(x_seasonality, 0), self.batch_size, axis = 0)
        return self
    
    def P(self, rho, z, z_pred):
        if z > z_pred:
            return rho*(z-z_pred)
        else:
            return (1-rho)*(z_pred-z)

if __name__ == "__main__":
    tf.reset_default_graph()
#    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:
        model = LSTM_SSM_model(sess, name = 'lesstightbounds', learning_rate = 0.0001, lr_decay = 0.999)
        model.build_LSTM().affine_transformations().build_model().build_loss().load_data()
        model.initialize_variables()
        loss = model.train(epochs = 5)
