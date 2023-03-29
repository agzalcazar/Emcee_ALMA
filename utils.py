from __future__ import division
import scipy.stats.kde as kde
import scipy.optimize as opt
import numpy as np
from scipy.optimize import minimize
from astropy.modeling.functional_models import Sersic2D
import arviz
from astropy.convolution import convolve, Gaussian2DKernel
import scipy.integrate as integrate


def zoom(image, dim):
    """
    select a square of NXN around the brightest pixel
    image: np.array of KxK dimensions
    dim: integer with dimensions of the square
    """
    max_rows, max_cols = np.where(image == np.nanmax(image))
    if dim % 2 == 0:
        dim = int(dim/2)
        max_rows, max_cols = np.where(image == np.nanmax(image))
        return image[max_rows[0] - dim: max_rows[0] + dim, max_cols[0] - dim: max_cols[0] + dim]
    else:
        dim = int(dim/2)
        return image[max_rows[0] - dim : max_rows[0] + dim + 1, max_cols[0] - dim : max_cols[0] + dim + 1]


def std_image(image):
    # square of 10x10 of a zoomed image. none effect of edges.
    return np.nanstd(image[0:10, 0:10].ravel())


def s_n(flux):
    max_signal = np.nanmax(flux)
    std = std_image(flux)
    return max_signal / std


def twoD_Gaussian_curvefit(grid, xo, yo, sigma_y, sigmax_minus_y, amplitude, theta):
    """
    :param grid: (x,y) of the meshgrid
    :param xo: x center of the gaussian
    :param yo: y center of the gaussian
    :param sigma_y:
    :param sigmax_minus_y: This parameters need to ensure that sigma_x >sigma_y and fix the orientation and angle
    :param amplitude:
    :param theta: angle os the gaussian
    :param offset:
    :return:
    """
    x, y = grid
    xo = float(xo)
    yo = float(yo)
    theta_rad = np.radians(theta)
    sigma_x = sigma_y + np.abs(
        sigmax_minus_y)  # on this way, I impose that sigma x is bigger than sigma y. Now angle is well define.

    a = (np.cos(theta_rad) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta_rad) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta_rad)) / (4 * sigma_x ** 2) + (np.sin(2 * theta_rad)) / (4 * sigma_y ** 2)
    c = (np.sin(theta_rad) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta_rad) ** 2) / (2 * sigma_y ** 2)
    g = amplitude * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))


    return g.ravel()


def sersic_curvefit(grid, amplitude_sersic, r_eff, x_0, y_0, ellip, angle_sersic):
    x, y = grid
    x_0 = float(x_0)
    y_0 = float(y_0)
    theta_rad = np.radians(180-angle_sersic) # Angle inverted, this is why i do 180-angle to follow the same criteria as gaussian.
    n = 1
    model = Sersic2D(amplitude_sersic, r_eff, n, x_0, y_0, ellip, theta_rad)

    return model(x, y).ravel()


def Gaussian_sersic(grid, x_0, y_0, sigma_y, sigmax_minus_y, amplitude, theta, amplitude_sersic, r_eff, ellip,
                    angle_sersic):
    x_0 = float(x_0)
    y_0 = float(y_0)
    model_sersic = sersic_curvefit(grid, amplitude_sersic, r_eff, x_0, y_0, ellip, angle_sersic)
    model_gaussian = twoD_Gaussian_curvefit(grid, x_0, y_0, sigma_y, sigmax_minus_y, amplitude, theta)
    return model_sersic + model_gaussian


def get_parameter_curve_fit(flux, x, y, max_flux, profile,beam):
    """
    :param flux: flux of image np.array NXN
    :param x: x component of the meshgrid
    :param y: y component of the meshgrid
    :return:parameters. It can be used as input for the MCMC code
    """
    bmin_s, bmaj_s = beam
    # # np.where(flux == np.max(flux))
    # cuadrante_1 = flux[16:34, 16:34]
    # cuadrante_2 = flux[0:16, 0:16]
    #
    # if sum(cuadrante_2.ravel() >= 0.4) > sum(cuadrante_1.ravel() >= 0.4):
    #     row, col = np.where(cuadrante_2 == np.max(cuadrante_2))
    #     r = np.sqrt(row ** 2 + col ** 2)
    #     angle = 180 - np.arcsin(col / r) * 180 / np.pi
    #
    # else:
    #     row, col = np.where(cuadrante_1 == np.max(cuadrante_1))
    #     r = np.sqrt(row ** 2 + col ** 2)
    #     angle = np.arcsin(col / r) * 180 / np.pi
    I = flux
    M0 = I.sum()
    x0 = (x * I).sum() / M0
    y0 = (y * I).sum() / M0
    Mxx = (x * x * I).sum() / M0 - x0 * x0
    Myy = (y * y * I).sum() / M0 - y0 * y0
    Mxy = (x * y * I).sum() / M0 - x0 * y0
    D = 2 * (Mxx * Myy - Mxy * Mxy)
    a = Myy / D
    c = Mxx / D
    b = -Mxy / D
    theta = np.degrees(0.5 * np.arctan(2 * c / (b - a)))
    if theta < 0:
        theta += 180

    if profile == 'A':
        initial_guess = (1, 0.3, 0, 0, 0.5, theta)
        popt, pcov = opt.curve_fit(sersic_curvefit, (x, y), flux.ravel(), p0=initial_guess,
                                   bounds=((0, 0, -0.2, -0.2, 0, 0),
                                           (1000, 1, 0.2, 0.2, 1, 180)))

    elif profile == 'B':
        # "x_0, y_0, sigma_y, sigmax_minus_y, amplitude, theta"

        initial_guess = (0, 0, 0.2, 0.2, 0.5, theta)
        popt, pcov = opt.curve_fit(twoD_Gaussian_curvefit, (x, y), flux.ravel(), p0=initial_guess,
                                   bounds=((-0.2, -0.2, 0, 0, 0.0001, 0.),
                                           (0.2,0.2,1,1,1000, 180)))

    elif profile == 'C':
        #" x_0, y_0, sigma_y, sigmax_minus_y, amplitude, theta, amplitude_sersic, r_eff, ellip, angle_sersic"
        initial_guess = (0, 0, bmin_s, bmaj_s-bmin_s, max_flux*0.6, theta, 0.1, 0.3, 0.5, theta)
        popt, pcov = opt.curve_fit(Gaussian_sersic, (x, y), flux.ravel(), p0=initial_guess,
                                   bounds=((-0.2, -0.2, 0, 0, 0.00001, 0.,0.00001,0,0,0),
                                           #(0.2, 0.2,bmin_s, bmaj_s-bmin_s, 1000, 180, 1000, 1, 1, 180)))
                                           (0.2, 0.2, 1, 1, max_flux, 180, 1000, 1, 1, 180)))

    return popt, pcov


def model_twoD_Gaussian(theta, x, y):
    """Function to fit, returns 2D gaussian function as 1D array
       theta; set of parameters to fit
       x,y: 1D np.arrays of dimension N

       return: height of the gaussian in for each pair (x,y)
    """

    amplitude, x0, y0, sigma_x, sigma_y, angle = theta

    theta_rad = np.radians(angle)

    a = (np.cos(theta_rad) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta_rad) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta_rad)) / (4 * sigma_x ** 2) + (np.sin(2 * theta_rad)) / (4 * sigma_y ** 2)
    c = (np.sin(theta_rad) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta_rad) ** 2) / (2 * sigma_y ** 2)
    g = amplitude * np.exp(- (a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2)))

    return g.ravel()


def model_Sersic(theta, x, y):
    """Function to fit, returns Sersic profile as 1D array
       theta; set of parameters to fit
       x,y: 1D np.arrays of dimension N

       return: Sersic profile for each pair (x,y)
    """

    amplitude_sersic, r_eff, x_0, y_0, ellip, angle_sersic = theta

    theta_rad = np.radians(180-angle_sersic)
    model = Sersic2D(amplitude_sersic, r_eff, 1, x_0, y_0, ellip, theta_rad)
    sersic = model(x, y)
    # if np.any(np.isnan(sersic)):
    #     sersic[:] = 0

    return sersic.ravel()

def log_likelihood(theta, x, y, z, zerr,bmin, bmaj,bpa,dy, profile):
    """
    Log-Likelihood of the model
    x,y: 1D np.arrays of dimension N (meshgrid)
    Z: 1D np.arrays of dimension N (flux)
    Zerr:
    Model:
        Model A corresponds to Sersic profile only
        Model B corresponds to Gaussian profile only
        Model C correspond to Model A + Model B
    """

    if profile == 'A':

        model = model_Sersic(theta, x, y)

    elif profile == 'B':

        model = model_twoD_Gaussian(theta, x, y)

    elif profile == 'C':

        amplitude, x_0, y_0, sigma_x, sigma_y, angle, amplitude_sersic, r_eff, ellip, angle_sersic = theta
        theta1 = [amplitude, x_0, y_0, sigma_x, sigma_y, angle]
        theta2 = [amplitude_sersic, r_eff, x_0, y_0, ellip, angle_sersic]

        gauss_point = model_twoD_Gaussian(theta1, x, y).reshape(35,35)
        # beam_gauss_kernel = Gaussian2DKernel(bmaj /(dy*3600),bmin /(dy*3600),theta=np.radians(bpa-90),x_size=35)
        # astropy_conv = convolve(gauss_point, beam_gauss_kernel)

        # model = astropy_conv.ravel() + model_Sersic(theta2, x, y)
        model = gauss_point.ravel() + model_Sersic(theta2, x, y)

    return -np.power((z - model), 2) / (2 * zerr ** 2) - 0.5 * np.log(2 * np.pi * zerr ** 2)


def lnprior(theta, curve_fit_mu_sigma, bmin, bmaj, max_flux, profile, ):
    """

    :param theta:
    :param curve_fit_mu_sigma:
    :param profile:
    :return:
    """
    if profile == 'A':

        amplitude_sersic, r_eff, x_0, y_0, ellip, angle_sersic = theta
        if not np.all(amplitude_sersic > 0 and r_eff > 0 and 0 <= ellip <= 1 and 0 <= angle_sersic <= 180 and  -0.2<x_0<0.2 and -0.2<y_0<0.2):
            return -np.inf
        # mu1 = 17  # curve_fit_mu_sigma[0]
        # # sigma1 = curve_fit_mu_sigma[1] * 10
        # sigma1 = 2
        # mu2 = 17  # curve_fit_mu_sigma[2]
        # sigma2 = 2  # curve_fit_mu_sigma[3] * 10
        # prior_x0 = np.log(1.0 / (np.sqrt(2 * np.pi) * sigma1)) - 0.5 * (x_0 - mu1) ** 2 / sigma1 ** 2
        # prior_y0 = np.log(1.0 / (np.sqrt(2 * np.pi) * sigma2)) - 0.5 * (y_0 - mu2) ** 2 / sigma2 ** 2
        # return prior_x0 + prior_y0
        return 0

    elif profile == 'B':

        amplitude, x_0, y_0, sigma_x, sigma_y, angle = theta
        if not np.all( 0 < amplitude < max_flux and sigma_x > 0 and 0 < sigma_y <= sigma_x and 0 <= angle <= 180 and -0.2<x_0<0.2 and -0.2<y_0<0.2):
            return -np.inf
        # mu1 = 17  # curve_fit_mu_sigma[0]
        # # sigma1 = curve_fit_mu_sigma[1] * 10
        # sigma1 = 2
        # mu2 = 17  # curve_fit_mu_sigma[2]
        # sigma2 = 2  # curve_fit_mu_sigma[3] * 10
        # prior_x0 = np.log(1.0 / (np.sqrt(2 * np.pi) * sigma1)) - 0.5 * (x_0 - mu1) ** 2 / sigma1 ** 2
        # prior_yo = np.log(1.0 / (np.sqrt(2 * np.pi) * sigma2)) - 0.5 * (y_0 - mu2) ** 2 / sigma2 ** 2
        # return prior_x0 + prior_yo
        return 0

    elif profile == 'C':
        amplitude, x_0, y_0, sigma_x, sigma_y, angle, amplitude_sersic, r_eff, ellip, angle_sersic = theta
        if not np.all(
                0 < amplitude < max_flux and -0.2 < x_0 < 0.2 and -0.2 < y_0 < 0.2 and 0 < sigma_x <= 2*bmaj and sigma_y <= sigma_x and 0 < sigma_y <= 2*bmin and 0 < angle <= 180 and
                amplitude_sersic > 0 and r_eff > sigma_x * 2.35 * 0.5 and 0 <= ellip <= 1 and 0 < angle_sersic <= 180):
                #0 < amplitude < max_flux and -0.2 < x_0 < 0.2 and -0.2 < y_0 < 0.2 and sigma_x>0 and 0 < sigma_y <= sigma_x  and 0 < angle <= 180 and
                # amplitude_sersic > 0 and r_eff > sigma_x * 2.35 * 0.5 and 0 <= ellip <= 1 and 0 < angle_sersic <= 180):
            return -np.inf
        # mu1 = 17.5  # curve_fit_mu_sigma[0]
        # #sigma1 = curve_fit_mu_sigma[1]
        # sigma1 = 2
        # mu2 = 17.5  # curve_fit_mu_sigma[2]
        # sigma2 = 2 #curve_fit_mu_sigma[3] * 1
        # prior_x0 = np.log(1.0 / (np.sqrt(2 * np.pi) * sigma1)) - 0.5 * (x_0 - mu1) ** 2 / sigma1 ** 2
        # prior_y0 = np.log(1.0 / (np.sqrt(2 * np.pi) * sigma2)) - 0.5 * (y_0 - mu2) ** 2 / sigma2 ** 2
        # prior_amplitude = cauchy.logpdf(amplitude,loc=curve_fit_mu_sigma[4], scale=0.1)
        # prior_amplitude_sersic = cauchy.logpdf(amplitude_sersic,loc=curve_fit_mu_sigma[5], scale=0.1)
        # prior_sigma_x = cauchy.logpdf(sigma_x, loc=curve_fit_mu_sigma[6], scale=0.1)
        # prior_sigma_y = cauchy.logpdf(sigma_x, loc=curve_fit_mu_sigma[7], scale=0.1)



        #return prior_x0 + prior_y0 + prior_amplitude_sersic + prior_amplitude+prior_sigma_x+prior_sigma_y
        return 0


def log_probability(theta, x, y, z, zerr, curve_fit_mu_sigma, bmin, bmaj,bpa, max_flux,dy, profile):#mirar como hacerlo mas compacto
    # log(likelihood * priors) = loglikelihood + log priors

    lp = lnprior(theta, curve_fit_mu_sigma, bmin, bmaj, max_flux,profile)
    if not np.isfinite(lp):
        return -np.inf#, -np.inf * np.ones(34*34)
    log_likelihood_ = log_likelihood(theta, x, y, z, zerr, bmin, bmaj,bpa,dy,profile)
    # if not np.isfinite(sum(log_likelihood_)):
    #     return -np.inf, -np.inf * np.ones(34*34)
    return lp + sum(log_likelihood_)#, log_likelihood_


def hpd_grid(sample, alpha=0.05, roundto=2):
    """Calculate highest posterior density (HPD) of array for given alpha.
    The HPD is the minimum width Bayesian credible interval (BCI).
    The function works for multimodal distributions, returning more than one mode
    Parameters
    ----------

    sample : Numpy array or python list
        An array containing MCMC samples
    alpha : float
        Desired probability of type I error (defaults to 0.05)
    roundto: integer
        Number of digits after the decimal point for the results
    Returns
    ----------
    hpd: list with the highest density interval
    x: array with grid points where the density was evaluated
    y: array with the density values
    modes: list listing the values of the modes

    """
    sample = np.asarray(sample)
    sample = sample[~np.isnan(sample)]
    # get upper and lower bounds
    l = np.min(sample)
    u = np.max(sample)
    density = kde.gaussian_kde(sample)
    x = np.linspace(l, u, 2000)
    y = density.evaluate(x)
    # y = density.evaluate(x, l, u) waitting for PR to be accepted
    xy_zipped = zip(x, y / np.sum(y))
    xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)
    xy_cum_sum = 0
    hdv = []
    for val in xy:
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= (1 - alpha):
            break
    hdv.sort()
    diff = (u - l) / 20  # differences of 5%
    hpd = []
    hpd.append(round(min(hdv), roundto))
    for i in range(1, len(hdv)):
        if hdv[i] - hdv[i - 1] >= diff:
            hpd.append(round(hdv[i - 1], roundto))
            hpd.append(round(hdv[i], roundto))
    hpd.append(round(max(hdv), roundto))
    ite = iter(hpd)
    hpd = list(zip(ite, ite))
    # modes = []
    # for value in hpd:
    #     x_hpd = x[(x > value[0]) & (x < value[1])]
    #     y_hpd = y[(x > value[0]) & (x < value[1])]
    #     # modes.append(round(x_hpd[np.argmax(y_hpd)], roundto))
    return hpd, x, y


def bic(parameters, x, y, flux, profile):
    """
    Returns the BIC score of a model.
    Input:-
    initialy: list of initial values of the parameters.
    y_pred: predicted values of y from a regression model in shape of an array
    p: number of variables used for prediction in the model.
    Output:-
    score: It outputs the BIC score type = int
    """

    # nll = lambda *args: -sum(log_likelihood(*args))
    # initial = np.array(initial)
    # soln = minimize(nll, initial, args=(x.ravel(), y.ravel(), flux.ravel(), std_image(flux)))
    # MLE_ll = sum(log_likelihood(soln.x, x.ravel(), y.ravel(), flux.ravel(), std_image(flux)))
    MLE_ll = sum(log_likelihood(parameters, x.ravel(), y.ravel(), flux.ravel(), std_image(flux), profile))
    p = len(parameters)
    BIC = p * np.log(x.shape[0] * x.shape[1]) - 2 * MLE_ll
    return BIC

def waic_(sampler,var_names):
    """
    :param sampler: emcee sampler
    :return: WAIC score and Pwaic
    """
    data = arviz.from_emcee(sampler,var_names=var_names)
    like_samps_flat = sampler.get_blobs(flat=True)['log_likelihood_']
    like_samps = np.hstack(like_samps_flat).reshape(sampler.chain.shape[0], sampler.chain.shape[1], 34 * 34)
    del(like_samps_flat)

    # for i in range(like_samps.shape[0]):
    #     for j in range(like_samps.shape[1]):
    #         like_samps_flat[i][j][:] = like_samps[i][j][0]

    data.add_groups(log_likelihood={"log likelihood": like_samps})
    del(like_samps)
    data = data.sel(draw=slice(2000, None))

    return arviz.waic(data)


def Tflux(theta, dx, profile, rms, points=100):
    x_grid = np.linspace(-1, 1, 8000)
    y_grid = np.linspace(-1, 1, 8000)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    diff = np.diff(np.linspace(-1, 1, 8000)[0:2])[0]

    if profile == 'B':
        flux = model_twoD_Gaussian(theta, x_grid, y_grid)
        integral = np.sum(flux * diff**2 / (dx*3600)**2) #change the grid dive by dx i change to mJy/pixel and multiplying by diff i select the new size of the pixel
    elif profile == 'A':
        flux = model_Sersic(theta, x_grid, y_grid)
        integral = np.sum(flux * diff ** 2 / (dx * 3600) ** 2)
        
    elif profile == 'C':

        amplitude, x_0, y_0, sigma_x, sigma_y, angle, amplitude_sersic, r_eff, ellip, angle_sersic = theta
        theta1 = [amplitude, x_0, y_0, sigma_x, sigma_y, angle]
        theta2 = [amplitude_sersic, r_eff, x_0, y_0, ellip, angle_sersic]        
        flux = model_twoD_Gaussian(theta1, x_grid, y_grid) + model_Sersic(theta2, x_grid, y_grid)
        integral = np.sum(flux * diff ** 2 / (dx * 3600) ** 2)

    integral_noise = []
    for i in range(0, points):
        noise = np.random.normal(loc=0.0, scale=rms, size=flux.shape)

        flux_noise = flux + noise
        integral_noise.append(np.sum(flux_noise * diff**2 / (dx*3600)**2))

    return integral, np.std(integral_noise)
