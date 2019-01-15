"""
TBD

Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
"""

import itertools
from numba import jit, njit, prange, vectorize, guvectorize, float64, cuda
import numpy as np
from math import sqrt, log, pi, exp
import time
import scipy
from joblib import Parallel, delayed
import multiprocessing

####################################
### Helper Fxns
####################################

@njit
def squash_x(x, a, b):
	#return 1-exp_parallel(-a*x)
	return 1/(1+exp_parallel(-a*x + b))

@njit
def squash_Wz(w, z):
	z = np.array([float(i) for i in z])
	return 1/(1+exp(-np.dot(w, z)))

@njit
def get_squashed_X(X, a, b):
	U = np.empty(X.shape)
	for img_ix in range(X.shape[0]):
		U[img_ix] = squash_x(X[img_ix, :], a, b)
	return U

@njit
def scaled_dot(u,v):
	return (np.dot(u,v)**2)**.5 / np.sum(u**2) / np.sum(v**2)

#@guvectorize([(float64[:], int64[:], float64[:])], '(n),(n)->(n)')
def squash_Wz_vec(w, z, res):
	for i in range(len(w)):
		res[i] = 1/(1+exp(-8*w[i]*z[i] + 4))

@njit
def exp_parallel(A):
	return np.array([exp(x) for x in A])

####################################
### Expectation-maximization
####################################

@njit
def get_all_p_x_z(mu, sigma, s_states, U, fixed_indices, z_states):
	#zeta=.1
	num_imgs = U.shape[0]
	num_units = U.shape[1]
	num_states = s_states.shape[0]
	num_features = s_states.shape[1]
	
	tmp = np.empty(num_units)
	p_x_z = np.zeros((num_imgs, num_states))
	
	for i_ix in range(num_imgs):
		for s_ix in range(num_states):
			tmp = exp_parallel( -( (U[i_ix, :] - mu - s_states[s_ix, :]) / sigma )**2 / 2) #/sigma
			p_x_z[i_ix, s_ix] = tmp.prod() * 100# ** zeta

		for f_ix in range(num_features):
			if i_ix in list(fixed_indices[f_ix, :]):
				for s_ix in range(num_states):
					if z_states[s_ix, f_ix] == 0:
						p_x_z[i_ix, s_ix] = 0

		p_x_z[i_ix, :] = p_x_z[i_ix, :] / np.amax(p_x_z[i_ix, :])
	
	return p_x_z

@njit
def get_p_z(z_states, theta):
	"""Symmetric Dirichlet prior with concentration ???"""

	zeta=1#.5
	num_states = len(z_states)
	num_features = len(z_states[0])

	p_z = np.ones(num_states)
	for s_ix in range(num_states):
		z = z_states[s_ix]

		for a in range(num_features):
			for b in range(a, num_features):
				p_z[s_ix] *= exp(theta[a,b]*z[a]*z[b])

		p_z[s_ix] = p_z[s_ix] ** zeta

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

def update_W(mu, W, z_states, U, p_z_x, fixed_indices, alpha, view_weights=False):
	num_states = z_states.shape[0]
	num_features = z_states.shape[1]
	num_imgs = U.shape[0]
	num_units = U.shape[1]
	W_est = np.zeros((num_features, num_units))
	pu2_i = np.zeros((num_units, num_states))
	pu_i = np.zeros((num_units, num_states))
	p_i = np.zeros(num_states)

	#relevant_states = []
	#for s_ix in range(num_states):
	#	if np.sum(p_z_x[:, s_ix]) > .01:
	#		relevant_states.append(s_ix)

	for s_ix in range(num_states):
		for u_ix in range(num_units):
			pu2_i[u_ix, s_ix] = np.sum(p_z_x[:, s_ix] * (U[:, u_ix] - mu[u_ix])**2)
			pu_i[u_ix, s_ix] = np.sum(p_z_x[:, s_ix] * (U[:, u_ix] - mu[u_ix]))
		p_i[s_ix] = np.sum(p_z_x[:, s_ix])

	#print(np.amax(pu2_i), np.amin(pu2_i))

	#print(np.amax(pu_i), np.amin(pu_i))

	#print(np.amax(p_i), np.amin(p_i))

	#t = time.time()
	W_est = scipy.optimize.minimize(lambda w_opt: W_opt_func(w_opt, pu2_i, pu_i, p_i,
			z_states, W, fixed_indices, U), np.ravel(W),
			jac=True, bounds=tuple(itertools.repeat((-2, 10),num_features*num_units)))['x'] #, lower=-100., upper=500.
	W_est = W_est.reshape((num_features, num_units))

	if view_weights:
		tmp=0
		for v_ix in range(W.shape[1]):
			if v_ix != u_ix:
				tmp += scaled_dot(W_est[:, u_ix], W[:, v_ix])
		print(W_opt_func(np.ravel(W_est), pu2_i, pu_i, p_i, z_states, W, fixed_indices, U)[0])#, np.sum((W_est[:, u_ix]**2)**.3) * .1, tmp * 500 / W.shape[1])

	#print(time.time()-t, end="")

	#for f_ix in range(num_features):
	#	W_est[f_ix][abs(W_est[f_ix]) < np.amax(abs(W_est[f_ix])) / 50] = 0

	return W_est #W + (W_est - W) * alpha

@njit
def update_mus(mu, s_states, U, p_z_x, beta):
	num_states = len(s_states)
	num_imgs = U.shape[0]
	num_units = U.shape[1]
	mu_est = np.empty(num_units)

	for u_ix in range(num_units):
		num = 0
		for s_ix in range(num_states):
			num += np.sum(p_z_x[:, s_ix] * (U[:, u_ix] - s_states[s_ix, u_ix]))
		
		mu_est[u_ix] = num / np.sum(p_z_x)

	return mu + (mu_est - mu) * beta

@njit
def update_sigma(mu, sigma, s_states, U, p_z_x):
	num_units = U.shape[1]
	num_states = len(s_states)
	num = 0

	for u_ix in range(num_units):
		for s_ix in range(num_states):
			num += np.sum(p_z_x[:, s_ix] * (U[:, u_ix] - mu[u_ix] - s_states[s_ix, u_ix])**2)

	sigma_est = sqrt(num / np.sum(p_z_x))

	return sigma_est

def update_ab(mu, a, b, sigma, s_states, X, p_z_x, alpha):
	num_states = s_states.shape[0]
	num_imgs = p_z_x.shape[0]
	num_units = X.shape[1]

	#relevant_states = []
	#for i_ix in range(num_imgs):
	#	relevant_states.append([])
	#	for s_ix in range(num_states):
	#		if p_z_x[i_ix, s_ix] > np.amax(p_z_x[i_ix, :]) / 10:
	#			relevant_states[i_ix].append(s_ix)

	temp = scipy.optimize.minimize(lambda ab: ab_opt_func(ab[0], ab[1], X, p_z_x, mu, s_states), (a,b), bounds=((1, 10), (2, 10)))['x']

	a_est, b_est = temp[0], temp[1]

	return a + (a_est - a) * alpha, b + (b_est - b) * alpha

####################################
### Objective fxns
####################################

@njit
def ab_opt_func(a, b, X, p_z_x, mu, s_states):
	ret = 0
	for i_ix in range(p_z_x.shape[0]):
		for s_ix in range(p_z_x.shape[1]):
			if p_z_x[i_ix, s_ix] > np.amax(p_z_x[i_ix, :]) / 10:
				ret += np.sum(p_z_x[i_ix, s_ix] * (squash_x(X[i_ix, :], a, b) - mu - s_states[s_ix, :])**2)

	return ret

@njit
def W_opt_func(w, a, b, c, z_states, W, fixed_indices, U):
	num_features = W.shape[0]
	num_units = W.shape[1]
	num_states = len(z_states)

	reg_norm = .1
	reg_dist = 10 / num_features
	reg_anno = 500 / W.shape[1]

	ret = 0
	#w[f_ix,:] = w[f_ix*num_units:(f_ix+1)*num_units]
	#w[:,u_ix] = w[u_ix::num_units]
	for f_ix in num_features:
		ret += np.sum((w[f_ix*num_units:(f_ix+1)*num_units]**2)**.3) * reg_norm

		jac[f_ix, :] += .6*reg_norm*w[f_ix*num_units:(f_ix+1)*num_units]**.6/abs(w[f_ix*num_units:(f_ix+1)*num_units])
		
		#for g_ix in num_features:
		#	if g_ix != f_ix:
		#		ret += scaled_dot(w[f_ix], W[g_ix]) * reg_dist

		#for i_ix in fixed_indices[f_ix]:
		#	ret += scaled_dot(w, U[i_ix, :]) * reg_anno

	jac = np.zeros((num_features, num_units))
	for u_ix in range(num_units):
		for s_ix in range(num_states):
			z = np.array([float(i) for i in z_states[s_ix]])
			wz = exp(-np.dot(w[u_ix::num_units], z))
			fwz = 1/(1+wz)
			#wz = squash_Wz(w[u_ix::num_units], z[s_ix])
			ret += a[u_ix, s_ix] - 2*b[u_ix, s_ix] * fwz + c[s_ix] * fwz**2

			for f_ix in range(num_features):
				if z_states[s_ix, f_ix]:
					jac[f_ix, u_ix] += -wz*(b[u_ix, s_ix] - c[s_ix]*fwz) / (wz + 1)**2

	return ret, np.ravel(jac)

@njit
def opt_func_gpu(w, a, b, c, z, relevant_states, W, fixed_indices, U):
	#num_features = W.shape[0]
	num_units = W.shape[1]
	num_states = len(relevant_states) #len(z)

	ret = np.array(num_units, num_states)
	threadsperblock = (16, 16)
	blockspergrid_x = math.ceil(num_units / threadsperblock[0])
	blockspergrid_y = math.ceil(num_states / threadsperblock[1])
	blockspergrid = (blockspergrid_x, blockspergrid_y)
	W_kernel[blockspergrid, threadsperblock](W, z, a, b, c, ret)

	#for u_ix in range(num_units):
	#	for s_ix in relevant_states:
	#		wz = squash_Wz(w[u_ix::num_units], z[s_ix])
	#		ret += a[u_ix, s_ix] + b[u_ix, s_ix] * wz + c[s_ix] * wz**2

	return ret

#@cuda.jit
"""def W_kernel(W, z, a, b, c, ret):
	sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
	sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
	sC = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
	sW = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
	sZ = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

	u_ix, s_ix = cuda.grid(2)

	if u_ix < ret.shape[0] and s_ix < ret.shape[1]:
		wz = 0
		for f_ix in range(len(z[0])):
			if z[s_ix, f_ix]==1:
				wz += W[u_ix+f_ix*num_units]
		wz = 1/(1+exp(-8*wz + 4))
		#wz = squash_Wz(W[u_ix::num_units], z[s_ix])
		ret[u_ix, s_ix] = a[u_ix, s_ix] + b[u_ix, s_ix] * wz + c[s_ix] * wz**2
"""
