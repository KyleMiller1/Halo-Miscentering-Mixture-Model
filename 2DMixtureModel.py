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
    Transforms a parameter from the MultiNest native space, [0, 1], to a physical value on the uniform distribution, [x1, x2].
    (See Section 5.1 of arXiv:0809.3437.)

    Parameters
    ----------
    c: float
        Parameter value in MultiNest native space, on [0, 1]
    x1: float
        Physical uniform distribution minimum
    x2: float
        Physical uniform distribution maximum

    Returns
    -------
    float of parameter's physical value, according to specified uniform distribution
    """
    return x1+c*(x2-x1)

def gaussian_prior(c, mu, sigma):
    """
    Transforms a parameter from the MultiNest native space, [0, 1], to a physical value on the Gaussian distribution, N(mu, sigma).
    (See Section 5.1 of arXiv:0809.3437.)

    Parameters
    ----------
    c: float
        Parameter value in MultiNest native space, on [0, 1]
    mu: float
        Physical Gaussian distribution mean
    sigma: float
        Physical Gaussian distribution standard deviation

    Returns
    -------
    float of parameter's physical value, according to specified Gaussian distribution
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
def proj_rho_D22(theta, r, lmax=40, nz=50):
    """
    Definition of the orbiting term of the halo profile model from Diemer 2022, projected to 2D.
    (See arXiv:2205.03420.)

    Parameters
    ----------
    theta: Nparam*1 array
        5 D22 orbiting model parameters in the form
            [log(alpha), log(beta), log(rho_s), log(r_s), log(r_t)]
    r: Nbins*1 array
        Radial bin midpoints (in R200m) at which to compute the model
    lmax: float
        Maximum line-of-sight distance of integration, in R200m
    nz: integer
        Number of projection integral samples to use

    Returns
    -------
    N*1 array of density profile values (in (Mpc/h)^-2)
    """

    # Unpack element-by-element so multinest doesn't complain
    lg_alpha = theta[0]
    lg_beta = theta[1]
    lg_rho_s = theta[2]
    lg_r_s  = theta[3]
    lg_r_t = theta[4]

    alpha = 10.**lg_alpha
    beta = 10.**lg_beta
    r_s = 10.**lg_r_s
    r_t = 10.**lg_r_t
    rho_s = 10**lg_rho_s

    def rho0_orbit(r):
        exp_arg = -(2/alpha)*((r/r_s)**alpha - 1) - (1/beta)*((r/r_t)**beta - (r_s/r_t)**beta)
        return rho_s*np.exp(exp_arg)

    z = np.logspace(np.log10(0.001), np.log10(lmax), nz)
    R_grid, z_grid = np.meshgrid(r,z,indexing='ij')
    r_grid = np.sqrt(R_grid**2 + z_grid**2)

    return 2.* integrate.simps(rho0_orbit(r_grid), z)

def rho_mis_given_r_mis(theta, r, r_mis, nz=50, phi_samples=100):
    """
    Miscentering correction model for an individual halo profile.
    (See arXiv:1811.06081.)

    Parameters
    ----------
    theta: Nparam*1 array
        5 D22 orbiting + 2 miscentered model parameters in the form
            [log(alpha), log(beta), log(rho_s), log(r_s), log(r_t), f_mis, sigma_r]
    r: Nbins*1 array
        Radial bin midpoints (in R200m) at which to compute the model
    r_mis: float
        Magnitude (in R200m) by which the halo is miscentered.
    nz: integer
        Number of integral samples to use in projecting D22 to 2D
    phi_samples: integer
        Number of np.trapz samples when integrating phi \in [0, 2*pi].

    Returns
    -------
    N*1 array of density profile values (in (Mpc/h)^-2)
    """

    def sub_integrand(phi, r, r_mis):
        return 1/(2*np.pi) * proj_rho_D22(theta[0:5], np.sqrt(r**2 + r_mis**2 + 2*r*r_mis*np.cos(phi)), nz=nz)

    def rho_mis_given_r_mis(r, r_mis):     
        phi = np.linspace(0, 2*np.pi, phi_samples)
        return np.array([np.trapz(sub_integrand(phi, r_i, r_mis), phi) for r_i in r])          

    return rho_mis_given_r_mis(r, r_mis)

#**************
# Fitting code
#**************
def fit_mixture_model(rvals, rhovals, covmats, base_path, out_dir=None, nz=50, rmis_samples=60, phi_samples=100):
    """
    Function to fit the mixture model to a collection of individual halo radial profile data vectors.

    Parameters
    ----------
    rvals: Nbins*1 array
    	Radial bin midpoints (in R200m) of the input profiles
    rhovals: Nhalos*Nbins array
	Radial surface number densities (in (Mpc/h)^-2) of the input profiles
    covmats: Nhalos*Nbins*Nbins array
    	Jackknife covariance matrix estimate of each input profile
    out_dir: str
        String specifying where to store chains from model fitting
    nz: integer
        Number of integral samples in projecting D22 to 2D
    rmis_samples: integer
        Number of r_mis samples over Rayleigh distribution in calculating prob_given_mis_center
    phi_samples: integer
        Number of np.trapz samples in integrating phi \in [0, 2*pi] in calculating rho_mis_given_r_mis
    """

    num_halos = len(rhovals)

    #****************************************
    # Likelihood and prior definitions
    # -> Formatted for emcee and pymultinest
    #****************************************
    def ln_like3d(theta, rvals, rhovals, covmats):
        """
        Gaussian ln(likelihood) definition used for fitting the mixture model.

        Parameters
        ----------
        theta: Nparam*1 array
            Mixture model parameters in the form 
            [log(alpha), log(beta), log(rho_s), log(r_s), log(r_t), f_mis, sigma_r]
        rvals: Nbins*1 array
    	    Radial values (bin midpoints in R200m) of the input profiles
        rhovals: Nhalos*Nbins array
	    Radial surface number densities (in (Mpc/h)^-2) of the input profiles
        covmats: Nhalos*Nbins*Nbins array
    	    Jackknife covariance matrix estimate of each input profile

        Returns
        -------
            Float of the likelihood for given model parameters and data vector
        """
        # *****************
        # Parameter Check:
        # *****************
        f_mis = theta[5]
        sigma_r = theta[6]
        if (f_mis < 0.0 or f_mis > 1.0 or sigma_r <= 0.0):
            return -np.inf

        # ***********************
        # Log_Prior Calculation:
        # ***********************
        # (Currently ignored to emphasize the likelihood.)
        # (Note that uniform_prior and gaussian_prior still affect the MultiNest physical parameter sampling.)

        log_prior = 0

        # ****************************
        # Log_Likelihood Calculation:
        # ****************************
        overall_log_likelihood = 0

        # Compute the theory predictions for both models.
        r_data = rvals[0]            
        rho_thr_D22 = proj_rho_D22(theta, r_data, nz=nz) 

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
            diff_from_misc = rhovals - rho_mis_given_r_mis(theta, r_data, r_mis, nz=nz, phi_samples=phi_samples)

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

        r_mis_vals = np.linspace(0, 4*sigma_r, rmis_samples)
        halo_gaussian_diffs = np.array([integrand(r_mis_i) for r_mis_i in r_mis_vals]).T
        prob_given_mis_center = np.average(halo_gaussian_diffs, axis=1, weights=prob_r_mis(r_mis_vals))

        ##########################################
        # log(likelihood_i)
        ##########################################
        log_likelihoods = np.log((1 - f_mis) * prob_given_cor_center + f_mis * prob_given_mis_center)
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

        cube[0] = uniform_prior(cube[0], log10(0.03), log10(0.4))  # lg_alpha            (D22)
        cube[1] = uniform_prior(cube[1], log10(0.1), log10(10))    # lg_beta             (D22)
        cube[2] = uniform_prior(cube[2], -20, 20)                  # lg_rho_s            (D22)
        cube[3] = uniform_prior(cube[3], log10(0.01), log10(0.45)) # lg_r_s              (D22)
        cube[4] = uniform_prior(cube[4], log10(0.5), log10(10))    # lg_r_t              (D22)
        cube[5] = uniform_prior(cube[5], 0, 1)                     # f_mis               (Misc.)
        cube[6] = uniform_prior(cube[6], 0, 1)                     # sigma_r             (Misc.)
	    
        return cube

    def loglike(cube, ndim=7, nparam=7):        
        return ln_like3d(cube, rvals, rhovals, covmats)

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

    pymultinest.run(loglike, prior, 7, n_live_points=650,
                    outputfiles_basename=base_path, resume=False, verbose=True, evidence_tolerance=0.01,
                    sampling_efficiency=0.8, n_iter_before_update=100, max_iter=0,
                    dump_callback=dumper_callback)

#*****************************************************************************
# SCRIPT EXECUTABLE
#*****************************************************************************
# Load the halo data.
data_path = "/path/to/halo/numpy/data"
rvals = np.load(data_path + "rvals.npy")
rhovals = np.load(data_path + "rhovals.npy")
covmats = np.load(data_path + "covmats.npy")

# Run MultiNest.
multinest_basepath = "/path/to/dump/multinest/outputs"
fit_mixture_model(rvals, rhovals, covmats, multinest_basepath, out_dir=None, nz=50, rmis_samples=60, phi_samples=100)
