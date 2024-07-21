#*****************************************************************************
# SCRIPT DEFINITIONS
#*****************************************************************************
import numpy as np
import pymultinest
from numpy import log10
from math import sqrt
from scipy import integrate
from scipy.special import erfcinv

class TermColors:
    """
    Escape sequences for different colors to print to terminal.
    -> Example usage: print(TermColors.CGREEN+"This is green"+TermColors.CEND)
    """
    
    CEND      = '\33[0m'
    CBOLD     = '\33[1m'
    CITALIC   = '\33[3m'
    CURL      = '\33[4m'
    CBLINK    = '\33[5m'
    CBLINK2   = '\33[6m'
    CSELECTED = '\33[7m'

    CGREEN  = '\33[32m'


def uniform_prior(c, x1, x2):
	"""
 	Uniform prior for parameter sampling. 

   	Parameters:
    	-----------
     	c: Cube object from multinest sampler
      	x1: Scalar
           Minimum value
       	x2: Scalar
	   Maximum value
  	"""
    return x1+c*(x2-x1)

def gaussian_prior(c, mu, sigma):
	"""
 	Gaussian prior for parameter sampling.

  	Parameters:
   	-----------
    	c: Cube object from multinest sampler
     	mu: Scalar
      	   Median value of parameter
	sigma: Scalar
 	   One standard deviation from median
  	"""
    if (c <= 1.0e-16):
        return -1.0e32
    elif ((1.0-c) <= 1.0e-16):
        return 1.0e32
    else:
        return mu+sigma*sqrt(2.0)*erfcinv(2.0*(1.0-c))

#*********************
# Profile definitions
#*********************
def rho_D22(theta, r, nz):
    """
    Definition of the orbiting term of the halo profile model from Diemer 2022.

    Parameters
    ----------
    theta: Nparam*1 array
        5 D22 model parameters in the form
            [log(alpha), log(beta), log(rho_s/rho_m), log(r_s), log(r_t)]
    r: N*1 array
        Radial values (in R200m/h) at which to compute the profile

    Returns
    -------
    N*1 array of density profile values
    """
    
    # Unpack element-by-element so multinest doesn't complain
    lg_alpha = theta[0]
    lg_beta = theta[1]
    lg_rho_s_over_rho_m = theta[2]
    lg_r_s  = theta[3]
    lg_r_t = theta[4]

    
    alpha = 10.**lg_alpha
    beta = 10.**lg_beta
    r_s = 10.**lg_r_s
    r_t = 10.**lg_r_t
    rho_s_over_rho_m = 10**lg_rho_s_over_rho_m
    
    
    def rho0_orbit(r):
        exp_arg = -(2/alpha)*((r/r_s)**alpha - 1) - (1/beta)*((r/r_t)**beta - (r_s/r_t)**beta)
        return rho_s_over_rho_m*np.exp(exp_arg)



    z = np.logspace(np.log10(0.001), np.log10(39.), nz)
    R_grid, z_grid = np.meshgrid(r,z,indexing='ij')
    r_grid = np.sqrt(R_grid**2 + z_grid**2)
    
    return 2.* integrate.simps(rho0_orbit(r_grid), z)
    


def rho_mis_given_r_mis(theta, r, r_mis, nz, intSlices=100):
    """
    Miscentering correction model for an individual cluster profile (implicitly invokes D22 orbiting term).

    Parameters
    ----------
    theta: Nparam*1 array
        2 Miscentered + 5 D22 model parameters in the form
            [log(alpha), log(beta), log(rho_s/rho_m), log(r_s), log(r_t), f, ln_(sigma_r)]
    r: N*1 array
        Radial values (in R200m/h) at which to compute the profile
    r_mis: float
        Magnitude (in R200m/h) by which the cluster profile is believed to be miscentered.
    intSlices: int
        Number of np.trapz integration slices when integrating phi \in [0, np.pi].

    Returns
    -------
    N*1 array of density profile values
    """

    def sub_integrand(phi, r, r_mis):
        return 1/(2*np.pi) * rho_D22(theta[0:5], np.sqrt(r**2 + r_mis**2 + 2*r*r_mis*np.cos(phi)), nz)
    
    def rho_mis_given_r_mis(r, r_mis):     
        phi = np.linspace(0, 2*np.pi, intSlices)
        return np.array([np.trapz(sub_integrand(phi, r_i, r_mis), phi) for r_i in r])          
    
    return rho_mis_given_r_mis(r, r_mis)


#**************
# Fitting code
#**************

def fit_mixture_model(rvals, rhovals, covmats, base_path, nz, out_dir=None, verbose=True, resume=False, ev_tolerance=0.01, samp_eff=0.8,
                      update_iter=100, max_iter=0, n_live_points=600, rmisSlices=20, phiSlices=100):
    """
    Function to fit the mixture model to a collection of individual halo profile data vectors.

    Parameters
    ----------
    rvals: arr
    	Cluster radial bins, same for all halos
    
    rhovals: arr
	Radial surface number densities, shape is [(Number of halos),(number of radial bins)]
    
    covmats:  arr
    	Array of covariance matrices, length is number of halos, each element is the respective halo's covariance matrix

    out_dir: str
        String specifying where to store chains from model fitting
        
    rmisSlices: int
        # integration slices taken over the r_mis Rayleigh distribution,
        in calculating prob_given_mis_center for ln_like3d.
        
    phiSlices: int
        # integration slices taken over phi \in [0, np.pi],
        in calculating rho_mis_given_r_mis for ln_like3d.
    """
    
    num_clusters = len(rhovals)

    #****************************************
    # Likelihood and prior definitions
    # -> Formatted for emcee and pymultinest
    #****************************************
    def ln_like3d(theta, rvals, rhovals, covmats, nz, rmisSlices, phiSlices):

        num_clusters = len(rhovals)

        """
        Gaussian ln(likelihood) definition used for fitting the mixture model.

        Parameters
        ----------
        theta: Nparam*1 array
            Mixture model parameters in the form 
            [log(alpha), log(beta), log(rho_s/rho_m), log(r_s), log(r_t), f, ln_(sigma_r)]

        rvals: N_halos*N_bins numpy array of each halo's radial values.
        rhovals: N_halos*Nbins numpy array of each halo's rho values.
        covmats: N_halos*Nbins*Nbins numpy array of each halo's covariance matrix.

        Returns
        -------
            Float of the likelihood for given model parameter and data vector
        """
        # *****************
        # Parameter Check:
        # *****************

        f = theta[5]
        if (f < 0.0 or f > 1.0):
            return -np.inf


        # ***********************
        # Log_Prior Calculation:
        # ***********************

        # (Currently ignored to emphasize the likelihood.) OR will have tiny impact
        # (Note that the prior still affects the parameter values sampled by MultiNest.)

	#If priors on f and sigma_r are gaussian:
        ln_sgr = theta[6]
        log_prior = -0.5*(f - 0.22)**2/0.11**2 -0.5*(ln_sgr - np.log(0.3))**2/0.22**2

        # ****************************
        # Log_Likelihood Calculation:
        # ****************************

        overall_log_likelihood = 0

        # Compute the theory predictions for both models.
        r_data = rvals[0]            
        rho_thr_D22 = rho_D22(theta, r_data, nz) 

        sigma_r = np.exp(ln_sgr)

        def prob_r_mis(r_mis):
            return r_mis/(sigma_r)**2 * np.exp(-(r_mis)**2 /(2*sigma_r**2))           

        ##########################################
        # Prob(data_i | correctly-centered, theta)
        ##########################################

        diff_from_D22 = rhovals - rho_thr_D22

        # Up next: prob_given_cor_center = np.exp(-1/2 * np.dot(diff_from_D22, np.linalg.solve(all_covmats, diff_from_D22)))
        partial = np.linalg.solve(covmats, diff_from_D22)

        global id_count
        id_count = 0
        def helper_one(row):
            global id_count
            res = np.dot(row, diff_from_D22[id_count])

            id_count += 1
            return res

        prob_given_cor_center = np.exp(-1/2 * np.apply_along_axis(helper_one, axis=1, arr=partial))
        id_count = 0 # Reset id_count.

        ##########################################
        # Prob(data_i | mis-centered, theta)
        ##########################################

        def integrand(r_mis):                                       
            diff_from_misc = rhovals - rho_mis_given_r_mis(theta, r_data, r_mis, nz, intSlices=phiSlices)

            # Up next: gaussian_diff = np.exp(-1/2 * np.dot(diff_from_misc, np.linalg.solve(all_covmats, diff_from_misc)))
            partial = np.linalg.solve(covmats, diff_from_misc)

            global id_count
            def helper_two(row):
                global id_count
                res = np.dot(row, diff_from_misc[id_count])

                id_count += 1
                return res

            gaussian_diff = np.exp(-1/2 * np.apply_along_axis(helper_two, axis=1, arr=partial))
            id_count = 0 # Reset id_count.

            return gaussian_diff # Don't multiply by prob(r_mis) here. np's weighted average will account for it.

        r_mis_vals = np.linspace(0, 4*sigma_r, rmisSlices)

        halo_gaussian_diffs = np.array([integrand(r_mis_i) for r_mis_i in r_mis_vals]).T

        prob_given_mis_center = np.average(halo_gaussian_diffs, axis=1, weights=prob_r_mis(r_mis_vals))

        ##########################################
        # log(likelihood_i)
        ##########################################

        log_likelihoods = np.log((1 - f) * prob_given_cor_center + f * prob_given_mis_center)

        overall_log_likelihood = np.sum(log_likelihoods)

        if (np.isnan(overall_log_likelihood)==True):
            return -np.inf

        # Return log(posterior) \propto log(likelihood) + log(prior).
        return overall_log_likelihood + log_prior

    def prior(cube, ndim=7, nparam=7):
        """
        Prior on parameters. Can be changed if chains aren't converging.
        Gaussian_prior and uniform_prior are functions defined in
        profile_calculations/utilities/multinest_priors that transform elements
        on the interval [0, 1] to gaussian or uniform distributions with the
        specified prameters.
        """

        cube[0] = uniform_prior(cube[0], log10(0.03), log10(0.4)) # lg_alpha (Diemer '22)
        cube[1] = uniform_prior(cube[1], log10(0.1), log10(10))  # lg_beta (Diemer '22)
        cube[2] = uniform_prior(cube[2], -20, 20)                 # lg_rho_s_over_rho_m (Diemer '22)
        cube[3] = uniform_prior(cube[3], log10(0.01), log10(0.45)) # lg_r_s (Diemer '22)
        cube[4] = uniform_prior(cube[4], log10(0.5), log10(10))     # lg_r_t (Diemer '22)
        cube[5] = gaussian_prior(cube[5], 0.22, 0.11)             # f
        cube[6] = gaussian_prior(cube[6], np.log(0.3), 0.22)     #sigma_r
	#Uniform miscentering priors:
        #cube[5] = uniform_prior(cube[5], 0, 1)             # f     
        #cube[6] = uniform_prior(cube[6], 0, 1)       # sigma_r, NOTE: not in log form

        return cube

    def loglike(cube, ndim=7, nparam=7):        
        return ln_like3d(cube, rvals, rhovals, covmats, nz, rmisSlices, phiSlices)
    
    def dumper_callback(nSamples, nlive, nPar, physLive, posterior,
                        paramConstr, maxLogLike, logZ, logZerr, nullcontext):
        
        print("----------------------------------------------------------")
        print("nSamples: ", nSamples)
        print("nlive: ", nlive)
        print("nPar: ", nPar)
        print("physLive: ", physLive)    
        print("posterior: ", posterior)           
        print("paramConstr: ", paramConstr)
        print("maxLogLike: ", maxLogLike)
        print("logZ: ", logZ)
        print("logZerr: ", logZerr)
        print("nullcontext: ", nullcontext)
        print("----------------------------------------------------------")
        

    print(TermColors.CGREEN+"Fitting profiles with multinest"+TermColors.CEND)
    
    pymultinest.run(loglike, prior, 7, n_live_points=n_live_points,
                    outputfiles_basename=base_path, resume=resume, verbose=verbose, evidence_tolerance=ev_tolerance,
                    sampling_efficiency=samp_eff, n_iter_before_update=update_iter, max_iter=max_iter,
                    dump_callback=dumper_callback)
    
    # Save chains
    n_params = 7 
    multinest_analyzer = pymultinest.Analyzer(n_params, base_path)
    
    return multinest_analyzer


#*****************************************************************************
# SCRIPT EXECUTABLE
#*****************************************************************************

# Load the cluster data.

path = "/home2/mky/"
path2 = "f0s0/"
covmats = np.load(path + path2 + "Mbin3_covmats.npy")
rhovals = np.load(path + path2 + "Mbin3_rhovals.npy")
r = np.load(path + path2 + "Mbin3_rvals.npy")

num_clusters = rhovals.shape[0]
rs = [0]*num_clusters
for i in range(rhovals.shape[0]):
	rs[i] = r
	covmats[i] = np.diag(np.diag(covmats[i]))
rvals = rs

basepath = "/home2/mky/sum23/MultiNest_Samples/multinest_samples"

# Run MultiNest.
fit_Mbin3_miscData_mixtureModel = fit_mixture_model(rvals, rhovals, covmats, basepath, nz=50, max_iter=0, n_live_points=1000, rmisSlices=50, phiSlices=60)

# Output the Analyzer data.
data = fit_Mbin3_miscData_mixtureModel.get_data()
post = fit_Mbin3_miscData_mixtureModel.get_equal_weighted_posterior()
stats = fit_Mbin3_miscData_mixtureModel.get_stats()

#Save the Analyzer data.
np.save("~/sum23/Mbin3_data", data)
np.save("~/sum23/Mbin3_post", post)
np.save("~/sum23/Mbin3_stats", stats)
