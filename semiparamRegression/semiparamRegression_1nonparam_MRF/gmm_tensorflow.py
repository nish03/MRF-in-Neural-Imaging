import numpy as np
import tensorflow as tf
import edward as ed
from edward.models import (
    Categorical, Dirichlet, Empirical, InverseGamma,
    MultivariateNormalDiag, Normal, ParamMixture)


def gmm_tensorflow(K, N, D,TRAINING_STEPS,beta_nonparam):
    ed.set_seed(42)
    pi = Dirichlet(tf.ones(K))
    mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)
    sigmasq = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)
    x = ParamMixture(pi, {'loc': mu, 'scale_diag': tf.sqrt(sigmasq)},MultivariateNormalDiag,sample_shape=N)
    z = x.cat 
    T = 500  # number of MCMC samples
    qpi = Empirical(tf.get_variable("qpi/params", [T, K],initializer=tf.constant_initializer(1.0 / K)))
    qmu = Empirical(tf.get_variable("qmu/params", [T, K, D],initializer=tf.zeros_initializer()))
    qsigmasq = Empirical(tf.get_variable("qsigmasq/params", [T, K, D],initializer=tf.ones_initializer()))
    qz = Empirical(tf.get_variable("qz/params", [T, N],initializer=tf.zeros_initializer(),dtype=tf.int32)) 
    inference = ed.Gibbs({pi: qpi, mu: qmu, sigmasq: qsigmasq, z: qz},data={x: beta_nonparam})
    inference.initialize()
    sess = ed.get_session()
    tf.global_variables_initializer().run()
    t_ph = tf.placeholder(tf.int32, [])
    running_cluster_means = tf.reduce_mean(qmu.params[:t_ph], 0)
    for _ in range(inference.n_iter):
      info_dict = inference.update()
      inference.print_progress(info_dict)
      t = info_dict['t']
      if t % inference.n_print == 0:
        print("\nInferred cluster means:")
        print(sess.run(running_cluster_means, {t_ph: t - 1}))
    means = qmu.eval()
    return means



