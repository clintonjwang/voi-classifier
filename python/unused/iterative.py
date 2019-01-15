
###########################
### Iterative (E-M?) Parametric Model
###########################

def kl_div_norm(m1, sig1, m2, sig2, one_sided="none"):
	#returns kl(p,q) where p~N(m1,s1), q~N(m2,s2)
	ret = np.log(sig2/sig1) + (sig1**2+(m1-m2)**2)/(2*sig2**2) - .5
	if one_sided=="less":
		return ret * (m1 < m2)
	elif one_sided=="greater":
		return ret * (m1 > m2)
	else:
		return ret

def obtain_params(A):
	"""Returns mu and var for the normal distribution p(A|f) based on the annotated set
	- A (100*10) is the list of annotated image activations for a given feature
	- F () is the list of feature labels
	"""
	return np.linalg.lstsq(A, F)

def fit_ls(A, Theta):
	"""
	- A (100) is the list of activations
	- Theta (100*15) is the matrix linking F to A
	- Returns feature labels (15)
	"""
	return np.linalg.lstsq(Theta, np.expand_dims(A, axis=1))

def neg_log_like(A, f, mu, var):
	"""Returns negative log likelihood, -log( p(A|f;mu,var) )
	- A (100) is the list of activations
	- f (15) is the list of feature labels, either 1 or 0
	- mu and var (15*100) are the params of the normal dist p(A|f)
	"""
	prob_A = np.sum([f[i] * np.prod([norm.pdf(A[a], mu[i,a], var[i,a]) for a in range(len(A))]) for i in range(len(f))]) / np.sum(f)
	return -math.log( prob_A )

def get_distribution(feature, population_activations):
	"""Returns the set of feature labels f that minimizes -log( p(A|f;mu,var) )
	"""
	pass
