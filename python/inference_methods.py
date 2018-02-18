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
def get_all_p_x_z(mu, m, sigma, s, z_states, filter_results, fixed_indices):
	zeta=.1
	num_imgs = filter_results.shape[0]
	num_states = len(z_states)
	num_units = m.shape[0]
	num_features = len(z_states[0])
	
	tmp = np.empty(num_units)
	p_x_z = np.zeros((num_imgs, num_states))
	
	for img_ix in range(num_imgs):
		for state_ix in range(num_states):
			z = np.array([float(z) for z in z_states[state_ix]])

			for u_ix in range(num_units):
				stdev = sqrt(np.dot(sigma[:, u_ix]**2, z) + s[u_ix]**2)
				tmp[u_ix] = 1/stdev * exp(-((filter_results[img_ix, u_ix] - np.dot(mu[:, u_ix], z) - m[u_ix])/stdev)**2/2)

			p_x_z[img_ix, state_ix] = tmp.prod() ** zeta


		for f_ix in range(num_features):
			if img_ix in list(fixed_indices[f_ix, :]):
				for state_ix in range(num_states):
					if z_states[state_ix][f_ix] == 0:
						p_x_z[img_ix, state_ix] = 0

		p_x_z[img_ix, :] = p_x_z[img_ix, :] / np.amax(p_x_z[img_ix, :])
	
	return p_x_z

@njit
def get_all_p_z_x(p_x_z, p_z):
	num_imgs = len(p_x_z)

	p_z_x = np.empty(p_x_z.shape)
	for img_ix in range(p_x_z.shape[0]):

		Z = np.dot(p_x_z[img_ix, :], p_z)
		for state_ix in range(p_x_z.shape[1]):
			p_z_x[img_ix, state_ix] = p_x_z[img_ix, state_ix] * p_z[state_ix] / Z

	return p_z_x

@njit
def get_p_z(z_states, theta_i, theta_ij):
	"""Symmetric Dirichlet prior with concentration ???"""

	zeta=.03
	num_states = len(z_states)
	num_features = len(z_states[0])

	p_z = np.ones(num_states)
	for state_ix in range(num_states):
		z = z_states[state_ix]

		for a in range(num_features-1):
			p_z[state_ix] *= exp(theta_i[a]*z[a])

			for b in range(a+1, num_features):
				p_z[state_ix] *= exp(theta_ij[a,b]*z[a]*z[b])

		p_z[state_ix] = p_z[state_ix] ** zeta

	return p_z / np.sum(p_z)

@njit
def update_thetas(p_z_x_sum, z_states_bool, theta_i, theta_ij, alpha):
	yotta = 1
	num_states = len(z_states_bool)
	num_features = len(z_states_bool[0])
	theta_i_est = np.zeros(num_features)
	theta_ij_est = np.zeros((num_features-1, num_features))
	
	for f_ix in range(num_features):
		p_z_x_rat_i = 0
		zi_0_sum = 0
		zi_1_sum = 0

		for state_ix in range(num_states):
			z = z_states_bool[state_ix]
			if z[f_ix]:
				p_z_x_rat_i += p_z_x_sum[state_ix]

				tmp = 0
				for a in range(num_features):
					tmp += theta_i[a] * z[a]
					for b in range(a+1,num_features):
						tmp += theta_ij[a,b] * (z[a] and z[b])

				zi_1_sum += exp(tmp - theta_i[f_ix])

			else:
				tmp = 0
				for a in range(num_features):
					tmp += theta_i[a] * z[a]
					for b in range(a+1,num_features):
						tmp += theta_ij[a,b] * (z[a] and z[b])

				zi_0_sum += exp(tmp)

		p_z_x_rat_i /= np.sum(p_z_x_sum)

		theta_i_est[f_ix] = (log(p_z_x_rat_i) + log(zi_0_sum) - log(zi_1_sum) - log(1-p_z_x_rat_i)) * yotta


	for f_ix in range(num_features-1):
		for g_ix in range(f_ix+1, num_features):
			p_z_x_rat_i = 0
			zi_0_sum = 0
			zi_1_sum = 0

			for state_ix in range(num_states):
				z = z_states_bool[state_ix]
				if z[f_ix] and z[g_ix]:
					p_z_x_rat_i += p_z_x_sum[state_ix]

					tmp = 0
					for a in range(num_features):
						tmp += theta_i[a] * z[a]
						for b in range(a+1, num_features):
							tmp += theta_ij[a,b] * (z[a] and z[b])

					zi_1_sum += exp(tmp - theta_ij[f_ix, g_ix])

				else:
					tmp = 0
					for a in range(num_features):
						tmp += theta_i[a] * z[a]
						for b in range(a+1, num_features):
							tmp += theta_ij[a,b] * (z[a] and z[b])

					zi_0_sum += exp(tmp)

			p_z_x_rat_i /= np.sum(p_z_x_sum)

			theta_ij_est[f_ix, g_ix] = log(p_z_x_rat_i) + log(zi_0_sum) - log(zi_1_sum) - log(1-p_z_x_rat_i)

	theta_i = theta_i + (theta_i_est - theta_i) * alpha
	theta_ij = theta_ij + (theta_ij_est - theta_ij) * alpha
	return theta_i, theta_ij

@njit
def update_mus(mu, m, sigma, s, z_states, filter_results, p_z_x, beta):
	gamma = 1
	num_states = len(z_states)
	num_features = len(z_states[0])
	num_units = m.shape[0]
	num_imgs = filter_results.shape[0]
	mu_est = np.empty((num_features, num_units))

	for f_ix in range(num_features):
		state_indices = [state_ix for state_ix in range(num_states) if z_states[state_ix][f_ix]==1]
		
		for g_ix in range(num_features):
			np.dot(mu[f_ix, :], mu[g_ix, :])

		for u_ix in range(num_units):
			num = 0
			den = 0
				
			for state_ix in state_indices:
				z = np.array([float(z) for z in z_states[state_ix]])

				mean_adj = np.dot(mu[:, u_ix], z) + m[u_ix] - mu[f_ix, u_ix]
				var = np.dot(sigma[:, u_ix]**2, z) + s[u_ix]**2
				
				den += np.sum(p_z_x[:, state_ix])/var
			
				for img_ix in range(num_imgs):
					num += p_z_x[img_ix, state_ix]/var * (filter_results[img_ix, u_ix] - mean_adj)
			
			regul = 0
			for g_ix in range(num_features):
				if g_ix == f_ix:
					continue
				regul += mu[g_ix, u_ix] / sqrt(np.sum(mu[f_ix, :]**2) * np.sum(mu[g_ix, :]**2))

			mu_est[f_ix, u_ix] = num / den - gamma * regul / num_features

	return mu + (mu_est - mu) * beta

@njit
def update_ms(mu, m, sigma, s, z_states, filter_results, p_z_x, beta):
	num_units = m.shape[0]
	num_states = len(z_states)
	num_imgs = filter_results.shape[0]
	m_est = np.empty(num_units)

	for u_ix in range(num_units):
		num = 0
		den = 0
			
		for state_ix in range(num_states):
			z = np.array([float(z) for z in z_states[state_ix]])

			mean_adj = np.dot(mu[:, u_ix], z)
			var = np.dot(sigma[:, u_ix]**2, z) + s[u_ix]**2
			
			den += np.sum(p_z_x[:, state_ix])/var
		
			for img_ix in range(num_imgs):
				num += p_z_x[img_ix, state_ix]/var * (filter_results[img_ix, u_ix] - mean_adj)
			
		m_est[u_ix] = num / den

	return m + (m_est - m) * beta

#@vectorize(['f64(f8, f8, f8, f8, f8)'], target='parallel')
#def Q_S_S(Var, a_i, c_i, z):
#	return a_i / (np.dot(z, Var[:-1])+Var[-1]) + c_i * log((np.dot(z, Var[:-1])+Var[-1]) * 2*pi) / 2

@njit
def update_stdevs_approx(mu, m, sigma, s, z_states, filter_results, p_z_x, beta):
	num_units = m.shape[0]
	num_states = len(z_states)
	num_features = len(z_states[0])
	num_imgs = filter_results.shape[0]

	sigma_est = np.empty((num_features, num_units))
	s_est = np.empty(num_units)
	var_adj = np.zeros(num_states)
	a_i = np.zeros(num_states)
	c_i = np.zeros(num_states)

	for u_ix in range(num_units):
		tmp=np.zeros(num_imgs)
		for img_ix in range(num_imgs):
			tmp[img_ix] = np.sum(p_z_x[img_ix, :]) * filter_results[img_ix, u_ix]
		s_est[u_ix] = np.std(tmp)

		for f_ix in range(num_features):
			tmp=np.zeros(num_imgs)
			for img_ix in range(num_imgs):
				for state_ix in range(num_states):
					if z_states[state_ix][f_ix]==1:
						tmp[img_ix] += p_z_x[img_ix, state_ix]
				tmp[img_ix] *= filter_results[img_ix, u_ix]

			sigma_est[f_ix, u_ix] = np.std(tmp)
			
	return sigma + (sigma_est - sigma) * beta, s + (s_est - s) * beta

def update_stdevs(mu, m, sigma, s, z_states, filter_results, p_z_x):
	num_units = m.shape[0]
	num_states = len(z_states)
	num_features = len(z_states[0])
	num_imgs = filter_results.shape[0]

	sigma_est = np.empty((num_features, num_units))
	s_est = np.empty(num_units)
	var_adj = np.zeros(num_states)
	a_i = np.zeros(num_states)
	c_i = np.zeros(num_states)

	#num_cores = multiprocessing.cpu_count() - 1
	for u_ix in range(num_units):
		for state_ix in range(num_states):
			a_i[state_ix] = np.sum([p_z_x[img_ix, state_ix] * (filter_results[img_ix, u_ix] - \
					   np.dot(mu[:, u_ix], z_states[state_ix]) - m[u_ix])**2 for img_ix in range(num_imgs)]) / 2
			c_i[state_ix] = np.sum(p_z_x[:, state_ix])

		temp = scipy.optimize.minimize(\
					lambda Var: np.sum([a_i[state_ix] / (np.dot(z_states[state_ix],
					   Var[:-1])+Var[-1]) + c_i[state_ix] * log((np.dot(z_states[state_ix],
					   Var[:-1])+Var[-1]) * 2*pi) / 2 for state_ix in range(num_states)]),
					  np.concatenate([sigma[:, u_ix]**2, [s[u_ix]**2]]),
					  bounds=tuple(itertools.repeat((.001, 1),num_features)) + tuple([(.005, 10)]))
					  #tuple(itertools.repeat((.001, .25),num_features+1)))

		sigma_est[:, u_ix] = [sqrt(i) for i in temp['x'][:-1]]
		s_est[u_ix] = sqrt(temp['x'][-1])
	#Parallel(n_jobs=num_cores)(delayed(_update_stdevs)(mu, m, sigma, s, z_states,
	#		filter_results, p_z_x, sigma_est, s_est, u_ix) for u_ix in range(num_units))

	return sigma_est, s_est

def _update_stdevs(mu, m, sigma, s, z_states, filter_results, p_z_x, sigma_est, s_est, u_ix):
	for state_ix in range(num_states):
		a_i[state_ix] = np.sum([p_z_x[img_ix, state_ix] * (filter_results[img_ix, u_ix] - \
				   np.dot(mu[:, u_ix], z_states[state_ix]) - m[u_ix])**2 for img_ix in range(num_imgs)]) / 2
		c_i[state_ix] = np.sum(p_z_x[:, state_ix])

	temp = scipy.optimize.minimize(\
				lambda Var: np.sum([a_i[state_ix] / (np.dot(z_states[state_ix],
				   Var[:-1])+Var[-1]) + c_i[state_ix] * log((np.dot(z_states[state_ix],
				   Var[:-1])+Var[-1]) * 2*pi) / 2 for state_ix in range(num_states)]),
				  np.concatenate([sigma[:, u_ix]**2, [s[u_ix]**2]]),
				  bounds=tuple(itertools.repeat((.001, 1),num_features)) + tuple([(.005, 10)]))
				  #tuple(itertools.repeat((.001, .25),num_features+1)))

	sigma_est[:, u_ix] = [sqrt(i) for i in temp['x'][:-1]]
	s_est[u_ix] = sqrt(temp['x'][-1])