"""
TBD

Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
"""

import itertools
from numba import jit, njit, prange, vectorize, guvectorize, float64
import numpy as np
from math import sqrt, log, pi, exp
import time
import scipy
from joblib import Parallel, delayed
import multiprocessing

####################################
### Expectation-maximization
####################################

@njit
def squash_x(x, a):
	return 1/(1+exp_parallel(-a*x + 4))

@njit
def unsquash_u(u, a):
	return -1

@njit
def squash_Wz(w, z):
	z = np.array([float(i) for i in z])
	return 1/(1+exp(-8*np.dot(w, z) + 4))

@njit
def exp_parallel(A):
	return np.array([exp(x) for x in A])

@njit
def get_all_p_x_z(mu, sigma, a, s_states, X, fixed_indices, z_states):
	#zeta=.1
	num_imgs = X.shape[0]
	num_units = X.shape[1]
	num_states = s_states.shape[0]
	num_features = s_states.shape[1]
	
	tmp = np.empty(num_units)
	p_x_z = np.zeros((num_imgs, num_states))
	
	for i_ix in range(num_imgs):
		for s_ix in range(num_states):
			tmp = 1/sigma * exp_parallel( -( (squash_x(X[i_ix, :], a) - mu - s_states[s_ix, :]) / sigma )**2 / 2)
			p_x_z[i_ix, s_ix] = tmp.prod()# ** zeta

		#for f_ix in range(num_features):
		#	if i_ix in list(fixed_indices[f_ix, :]):
		#		for t_ix in range(num_states):
		#			if z_states[t_ix][f_ix] == 0:
		#				p_x_z[i_ix, t_ix] = 0

		p_x_z[i_ix, :] = p_x_z[i_ix, :] / np.amax(p_x_z[i_ix, :])
	
	return p_x_z

@njit
def get_p_z(z_states, theta):
	"""Symmetric Dirichlet prior with concentration ???"""

	#zeta=.03
	num_states = len(z_states)
	num_features = len(z_states[0])

	p_z = np.ones(num_states)
	for s_ix in range(num_states):
		z = z_states[s_ix]

		for a in range(num_features):
			for b in range(a, num_features):
				p_z[s_ix] *= exp(theta[a,b]*z[a]*z[b])

		#p_z[s_ix] = p_z[s_ix] ** zeta

	return p_z / np.sum(p_z)

@njit
def get_s_states(z_states, W, p_z):
	"""Symmetric Dirichlet prior with concentration ???"""

	#zeta=.03
	num_units = W.shape[1]
	num_states = len(z_states)
	num_features = len(z_states[0])

	s_states = np.empty((num_states, num_units))
	for s_ix in range(num_states):
		for u_ix in range(num_units):
			s_states[s_ix, u_ix] = squash_Wz(W[:, u_ix], z_states[s_ix])

	return s_states

@njit
def get_all_p_z_x(p_x_z, p_z):
	num_imgs = len(p_x_z)

	p_z_x = np.empty(p_x_z.shape)
	for img_ix in range(p_x_z.shape[0]):

		Z = np.dot(p_x_z[img_ix, :], p_z)
		for s_ix in range(p_x_z.shape[1]):
			p_z_x[img_ix, s_ix] = p_x_z[img_ix, s_ix] * p_z[s_ix] / Z

	return p_z_x

@njit
def update_thetas(p_z_x_sum, z_states_bool, theta, alpha):
	#yotta = 1
	num_states = len(z_states_bool)
	num_features = len(z_states_bool[0])
	theta_est = np.zeros((num_features, num_features))

	for f_ix in range(num_features):
		for g_ix in range(f_ix, num_features):
			p_z_x_rat_i = 0
			zi_0_sum = 0
			zi_1_sum = 0

			for s_ix in range(num_states):
				z = z_states_bool[s_ix]
				if z[f_ix] and z[g_ix]:
					p_z_x_rat_i += p_z_x_sum[s_ix]

					tmp = 0
					for a in range(num_features):
						for b in range(a,num_features):
							tmp += theta[a,b] * (z[a] and z[b])

					zi_1_sum += exp(tmp - theta[f_ix, g_ix])

				else:
					tmp = 0
					for a in range(num_features):
						for b in range(a,num_features):
							tmp += theta[a,b] * (z[a] and z[b])

					zi_0_sum += exp(tmp)

			p_z_x_rat_i /= np.sum(p_z_x_sum)

			theta_est[f_ix, g_ix] = log(p_z_x_rat_i) + log(zi_0_sum) - log(zi_1_sum) - log(1-p_z_x_rat_i)

	return theta + (theta_est - theta) * alpha


def squash_Wz_plain(w, z):
	return 1/(1+exp(-8*np.dot(w, z) + 4))

def update_W(mu, a, W, z_states, X, p_z_x, alpha):
	num_states = len(z_states)
	num_features = len(z_states[0])
	num_imgs = X.shape[0]
	num_units = X.shape[1]
	W_est = np.empty((num_features, num_units))
	a_i = np.zeros(num_states)
	b_i = np.zeros(num_states)

	for u_ix in range(num_units):
		for s_ix in range(num_states):
			a_i[s_ix] += np.sum(p_z_x[:, s_ix]**.5 * (squash_x(X[:, u_ix], a) - mu[u_ix]))
			b_i[s_ix] += np.sum(p_z_x[:, s_ix]**.5)

		W_est[:, u_ix] = scipy.optimize.minimize(lambda w: np.sum([a_i[s_ix] * squash_Wz(w, z_states[s_ix]) - b_i[s_ix] for s_ix in range(num_states)]), W[:, u_ix])['x']

	return W + (W_est - W) * alpha

@njit
def update_mus(mu, a, W, sigma, s_states, X, p_z_x, fixed_indices, beta):
	num_states = len(s_states)
	num_imgs = X.shape[0]
	num_units = X.shape[1]
	mu_est = np.empty(num_units)

	for u_ix in range(num_units):
		num = 0

		for s_ix in range(num_states):
			num += np.sum(p_z_x[:, s_ix] * (squash_x(X[:, u_ix], a) - s_states[s_ix, u_ix]))
		
		mu_est[u_ix] = num / np.sum(p_z_x)

	return mu + (mu_est - mu) * beta

@njit
def update_sigma(mu, a, sigma, s_states, X, p_z_x):
	num_units = X.shape[1]
	num_states = len(s_states)
	num = 0

	for u_ix in range(num_units):
		for s_ix in range(num_states):
			num += np.sum(p_z_x[:, s_ix] * (squash_x(X[:, u_ix], a) - mu[u_ix] - s_states[s_ix, u_ix])**2)

	sigma_est = sqrt(num / np.sum(p_z_x))

	return sigma_est

@njit
def update_a(mu, sigma, s_states, X, p_z_x):
	return None