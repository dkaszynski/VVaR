import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, t
from scipy.optimize import minimize


def get_dict_tests():
    """The dictionary of the implemented and available VaR backtests
    Parameters
    ----------
    Returns
    -------
    out : dictionary with test names (str) and references to functional forms (functions)
    ---------
    """

    out = {'Kupiec-POF': kupiec_pof,
           'Binomial-POF': binomial_pof,
           'Kupiec-TUFF': kupiec_tuff,
           'Christoffersen-ICoverage': christoffersen_icov,
           'Christoffersen-CCoverage': christoffersen_ccov,
           'Haas-TBF': haas_tbf,
           'Christoffersen-CWeibull': christoffersen_cweibull,
           'Haas-DWeibull': haas_dweibull,
           'Engle-DQ': engle_dq,
           'Berkowitz-BoxLjung': berkowitz_bl,
           'Kramer-GINI': kramer_gini}

    return out


def auto_corr(x, lag=1):
    return np.corrcoef(x[:-lag].T, x[lag:].T)[0, 1]


# Discrete Weibull distribution
def f_dw(d_i, a, b):
    return max((1 - a) ** np.power(d_i - 1, b) - (1 - a) ** np.power(d_i, b), 10**-11)


def F_dw(d_i, a, b):
    return max(1 - (1 - a) ** np.power(d_i, b), 10**-11)


def dw_log_likelihood(b, a, d, censored=None, sign=-1.0):
    if censored is None:
        out = [np.log(f_dw(d_i, a, b)) for d_i in d]
    else:
        if len(d) > 1:
            out = [np.log(f_dw(d_i, a, b)) for d_i in d[1:-1]]
            out.append(np.log(1 - F_dw(d[0], a, b)))
            out.append(np.log(1 - F_dw(d[-1], a, b)))
        else:
            out = sign * np.log(1 - F_dw(d, a, b))
            return out

    # For the -infy, take floor of precision
    out = sign * sum([np.floor(np.log(np.finfo(float).tiny)) if np.isneginf(i) else i for i in out])
    return out


# Continuous Weibull distribution
def f_cw(d_i, lam, k):
    return min(max(k / lam * np.power((d_i) / lam, k - 1) * np.exp(- np.power((d_i) / lam, k)), 10**-11), 1-10**-11)


def F_cw(d_i, lam, k):
    return min(max(1 - np.exp(- np.power((d_i) / lam, k)), 10**-11), 1-10**-11)


def cw_log_likelihood(params, d, censored=None, sign=-1.0):
    lam = params[0]
    k = params[1]

    if censored is None:
        out = [np.log(f_cw(d_i, lam, k)) for d_i in d]
    else:
        if len(d) > 1:
            out = [np.log(f_cw(d_i, lam, k)) for d_i in d[1:-1]]
            out.append(np.log(1 - F_cw(d[0], lam, k)))
            out.append(np.log(1 - F_cw(d[-1], lam, k)))
        else:
            out = sign * np.log(1 - F_cw(d, lam, k))
            return out

    # For the -inf, take floor of precision
    out = sign * sum([np.floor(np.log(np.finfo(float).tiny)) if np.isneginf(i) else i for i in out])
    return out


# Numerical hessian
def finite_hessian(f, x, h=10 ** -8):
    n = len(x)
    out = np.full_like(np.empty([n, n]), np.nan)
    di = np.eye(n)
    f_x = f(x)
    for i in range(n):
        e_i = di[i, :]
        for j in range(n):
            if i == j:
                out[i, j] = (f(x + e_i * h) - 2 * f_x + f(x - e_i * h)) / (h ** 2)
                continue
            e_j = di[:, j]
            out[i, j] = (f(x + e_i * h + e_j * h) - f(x - e_i * h + e_j * h) -
                         f(x + e_i * h - e_j * h) + f(x - e_i * h - e_j * h)) / (4 * h ** 2)
    return out


def kupiec_pof(y: pd.DataFrame = None,
               y_p: pd.DataFrame = None,
               h_s: pd.DataFrame = None,
               p: float = None,
               **kwargs: dict) -> dict:
    """Kupiec Proportion of Failures test
    Parameters
    ----------
    y   : numerical pd.DataFrame with empirical P&L
    y_p : numerical pd.DataFrame with Value at Risk with level of p
    h_s : boolean pd.DataFrame with hit series (alternative to y < y_p)
    p   : numerical scalar of Value at Risk level
    Returns
    -------
    out : dictionary with test statistics and p-value
    Reference
    ---------
    Kupiec, P., 1995. Techniques for verifying the accuracy of risk measurement models. The J. of Derivatives, 3(2).
    """

    if all(input_var is None for input_var in (y, y_p, h_s)):
        out = {'lr': None,
               'p-value': None}
        return out

    if h_s is None:
        h_s = y < y_p

    n = int(len(h_s.values))
    s = float(sum(h_s.values))

    lr = -2 * np.log((1 - p) ** (n - s) * p ** s) \
         + 2 * np.log((1 - s / n) ** (n - s) * (s / n) ** s)
    p_value = 1 - chi2.cdf(lr, 1)

    out = {'lr': lr,
           'p-value': p_value}
    return out


def binomial_pof(y: pd.DataFrame = None,
                 y_p: pd.DataFrame = None,
                 h_s: pd.DataFrame = None,
                 p: float = None,
                 **kwargs: dict) -> dict:
    """Binomial Proportion of Failures test
    Parameters
    ----------
    y   : numerical pd.DataFrame with empirical P&L
    y_p : numerical pd.DataFrame with Value at Risk with level of p
    h_s : boolean pd.DataFrame with hit series (alternative to y < y_p)
    p   : numerical scalar of Value at Risk level
    Returns
    -------
    out : dictionary with test statistics and p-value
    Reference
    ---------
    # Philippe, J., 2001. Value at risk: the new benchmark for managing financial risk. NY: McGraw-Hill Professional.
    """

    if all(input_var is None for input_var in (y, y_p, h_s)):
        out = {'lr': None,
               'p-value': None}
        return out

    if h_s is None:
        h_s = y < y_p

    n = int(len(h_s.values))
    s = float(sum(h_s.values))

    lr = (s - p * n) / (p * (1 - p) * n) ** (1 / 2)

    p_value = 2 * min(1 - norm.cdf(lr), norm.cdf(lr))

    out = {'lr': lr,
           'p-value': p_value}
    return out


def kupiec_tuff(y: pd.DataFrame = None,
                y_p: pd.DataFrame = None,
                h_s: pd.DataFrame = None,
                p: float = None,
                **kwargs: dict) -> dict:

    """Kupiec Time Until First Failure
    Parameters
    ----------
    y   : numerical pd.DataFrame with empirical P&L
    y_p : numerical pd.DataFrame with Value at Risk with level of p
    h_s : boolean pd.DataFrame with hit series (alternative to y < y_p)
    p   : numerical scalar of Value at Risk level
    Returns
    -------
    out : dictionary with test statistics and p-value
    Reference
    ---------
    Kupiec, P., 1995. Techniques for verifying the accuracy of risk measurement models. The J. of Derivatives, 3(2).
    """

    if all(input_var is None for input_var in (y, y_p, h_s)):
        out = {'lr': None,
               'p-value': None}
        return out

    if h_s is None:
        h_s = y < y_p

    n = int(len(h_s.values))
    tuff = np.where(h_s)[0] + 1

    # No exceeds scenario
    if len(tuff) == 0 and 1 / p < n:
        tuff = n + 1
    elif len(tuff) == 0:
        out = {'lr': None,
               'p-value': None}
        return out

    # Select first failure
    if not isinstance(tuff, int):
        tuff = tuff[0]

    tuff = int(tuff)

    counter = p * (1 - p) ** (tuff - 1)
    denominator = (1 / tuff) * (1 - (1 / tuff)) ** (tuff - 1)

    lr = -2 * np.log(counter / denominator)
    p_value = 1 - chi2.cdf(lr, 1)

    out = {'lr': lr,
           'p-value': p_value}
    return out


def christoffersen_icov(y: pd.DataFrame = None,
                        y_p: pd.DataFrame = None,
                        h_s: pd.DataFrame = None,
                        p: float = None,
                        **kwargs: dict) -> dict:
    """Christoffersen independence test
    Parameters
    ----------
    y   : numerical pd.DataFrame with empirical P&L
    y_p : numerical pd.DataFrame with Value at Risk with level of p
    h_s : boolean pd.DataFrame with hit series (alternative to y < y_p)
    p   : numerical scalar of Value at Risk level
    Returns
    -------
    out : dictionary with test statistics and p-value
    Reference
    ---------
    Christoffersen, P. 1998. Evaluating interval forecasts. International Economic Review 39:841–62.
    """

    if all(input_var is None for input_var in (y, y_p, h_s)):
        out = {'lr': None,
               'p-value': None}
        return out

    if h_s is None:
        h_s = y < y_p

    h_s = np.array(h_s, dtype=float)

    h_s_lag = h_s[1:] - h_s[:-1]

    n_01 = (h_s_lag == 1).sum()
    n_10 = (h_s_lag == -1).sum()
    n_11 = (h_s[1:][h_s_lag == 0] == 1).sum()
    n_00 = (h_s[1:][h_s_lag == 0] == 0).sum()

    if all((n_00 == 0, n_01 == 0)) or all((n_11 == 0, n_10 == 0)):
        out = {'lr': None,
               'p-value': None}
        return out

    n_0 = n_01 + n_00
    n_1 = n_10 + n_11
    n = n_0 + n_1
    p_01, p_11 = n_01 / (n_00 + n_01), n_11 / (n_11 + n_10)
    q = n_1 / n

    # Independence test
    ind_h0 = (n_00 + n_01) * np.log(1 - q) + (n_01 + n_11) * np.log(q)
    ind_h1 = n_00 * np.log(1 - p_01) + n_01 * np.log(p_01) + n_10 * np.log(1 - p_11)
    if p_11 > 0:
        ind_h1 += n_11 * np.log(p_11)

    lr = -2 * (ind_h0 - ind_h1)
    p_value = 1 - chi2.cdf(lr, 1)

    out = {'lr': lr,
           'p-value': p_value}
    return out


def christoffersen_ccov(y: pd.DataFrame = None,
                        y_p: pd.DataFrame = None,
                        h_s: pd.DataFrame = None,
                        p: float = None,
                        **kwargs: dict) -> dict:
    """Christoffersen conditional coverage test
    Parameters
    ----------
    y   : numerical pd.DataFrame with empirical P&L
    y_p : numerical pd.DataFrame with Value at Risk with level of p
    h_s : boolean pd.DataFrame with hit series (alternative to y < y_p)
    p   : numerical scalar of Value at Risk level
    Returns
    -------
    out : dictionary with test statistics and p-value
    Reference
    ---------
    Christoffersen, P. 1998. Evaluating interval forecasts. International Economic Review 39:841–62.
    """

    if all(input_var is None for input_var in (y, y_p, h_s)):
        out = {'lr': None,
               'p-value': None}
        return out

    if h_s is None:
        h_s = y < y_p

    h_s = np.array(h_s, dtype=float)

    h_s_lag = h_s[1:] - h_s[:-1]

    n_01 = (h_s_lag == 1).sum()
    n_10 = (h_s_lag == -1).sum()
    n_11 = (h_s[1:][h_s_lag == 0] == 1).sum()
    n_00 = (h_s[1:][h_s_lag == 0] == 0).sum()

    if all((n_00 == 0, n_01 == 0)) or all((n_11 == 0, n_10 == 0)):
        out = {'lr': None,
               'p-value': None}
        return out

    n_0 = n_01 + n_00
    n_1 = n_10 + n_11
    n = n_0 + n_1
    p_01, p_11 = n_01 / (n_00 + n_01), n_11 / (n_11 + n_10)
    q = n_1 / n

    # Unconditional coverage test
    uc_h0 = n_0 * np.log(1 - p) + n_1 * np.log(p)
    uc_h1 = n_0 * np.log(1 - q) + n_1 * np.log(q)
    uc = -2 * (uc_h0 - uc_h1)

    # Independence test
    ind_h0 = (n_00 + n_01) * np.log(1 - q) + (n_01 + n_11) * np.log(q)
    ind_h1 = n_00 * np.log(1 - p_01) + n_01 * np.log(p_01) + n_10 * np.log(1 - p_11)
    if p_11 > 0:
        ind_h1 += n_11 * np.log(p_11)
    ind = -2 * (ind_h0 - ind_h1)

    # Conditional coverage test
    cc = uc + ind

    lr = uc + cc
    p_value = 1 - chi2.cdf(lr, 2)

    out = {'lr': lr,
           'p-value': p_value}

    return out


def haas_tbf(y: pd.DataFrame = None,
             y_p: pd.DataFrame = None,
             h_s: pd.DataFrame = None,
             p: float = None,
             **kwargs: dict) -> dict:
    """Haas Time Between Failures
    Parameters
    ----------
    y   : numerical pd.DataFrame with empirical P&L
    y_p : numerical pd.DataFrame with Value at Risk with level of p
    h_s : boolean pd.DataFrame with hit series (alternative to y < y_p)
    p   : numerical scalar of Value at Risk level
    Returns
    -------
    out : dictionary with test statistics and p-value
    Reference
    ---------
    Haas, M., 2001. New methods in backtesting. Financial Engineering Research Center, Bonn.
    """

    if all(input_var is None for input_var in (y, y_p, h_s)):
        out = {'lr': None,
               'p-value': None}
        return out

    if h_s is None:
        h_s = y < y_p

    n = len(h_s.values)
    s = sum(h_s.values)

    tbf = np.where(h_s)[0] + 1

    if len(tbf) == 0 and 1 / p < n:
        tbf = n + 1
    elif len(tbf) == 0:
        out = {'lr': None,
               'p-value': None}
        return out

    tbf = np.diff(np.hstack((0, tbf)))

    def lr_i(a, v):
        return -2 * np.log((a * (1 - a) ** (v - 1)) /
                           ((1 / v) * (1 - 1 / v) ** (v - 1)))

    lr_ind = sum([lr_i(p, item) for item in tbf])

    lr_pof = -2 * np.log((1 - p) ** (n - s) * p ** s) \
             + 2 * np.log((1 - s / n) ** (n - s) * (s / n) ** s)

    lr = lr_ind + lr_pof
    p_value = 1 - chi2.cdf(lr, len(tbf) + 1)

    out = {'lr': lr[0],
           'p-value': p_value[0]}

    return out


def christoffersen_cweibull(y: pd.DataFrame = None,
                            y_p: pd.DataFrame = None,
                            h_s: pd.DataFrame = None,
                            p: float = None,
                            **kwargs: dict) -> dict:
    """Christoffersen & Pelletier Continuous Weibull test
    Parameters
    ----------
    y   : numerical pd.DataFrame with empirical P&L
    y_p : numerical pd.DataFrame with Value at Risk with level of p
    h_s : boolean pd.DataFrame with hit series (alternative to y < y_p)
    p   : numerical scalar of Value at Risk level
    Returns
    -------
    out : dictionary with test statistics and p-value
    Reference
    ---------
    Christoffersen, P. and Pelletier, D., 2004. Backtesting value-at-risk: A duration-based approach.
    Journal of Financial Econometrics, 2(1), pp.84-108.
    """

    if all(input_var is None for input_var in (y, y_p, h_s)):
        out = {'lr': None,
               'p-value': None}
        return out

    if h_s is None:
        h_s = (y < y_p)

    h_s = np.array(h_s, dtype=float)

    n = len(h_s)
    tbf = np.where(h_s)[0] + 1

    if len(tbf) < 1:
        out = {'lr': None, 'p-value': None}
        return out

    censored = np.repeat(0, len(tbf)).tolist()
    
    if h_s[0] == 0:
        censored[0] = 1
        tbf = np.hstack((1, tbf))
    if h_s[-1] == 0:
        censored[-1] = 1
        tbf = np.hstack((tbf, len(h_s)))

    tbf = np.diff(tbf)
        
    res = minimize(cw_log_likelihood, np.array([1/p, 2], dtype=float), args=(tbf, censored, -1),
                   method='L-BFGS-B', options={'disp': False}, bounds=((1/p-0.1, 1/p+0.1), (10**-8, None)))

    # Terminate if solver couldn't find solution
    if not res.success or res.x[1] < 10 ** (-8):
        out = {'lr': None, 'p-value': None}
        return out

    # Functional form of Continuous Weibull log-likelihood function
    #cw_llik_fun = lambda params: cw_log_likelihood(params, tbf, censored=censored, sign=-1)

    # Calculate numerical hessian of cw_log_likelihood fun in the point res.x
    #hess = finite_hessian(cw_llik_fun, res.x, h=10 ** (-4))

    # Check if hessian has been calculated
    #if np.linalg.matrix_rank(hess) < min(hess.shape):
    #    out = {'lr': None, 'p-value': None}
    #    return out

    # Calculate standard errors
    #se_matrix = np.diag(np.linalg.inv(hess))
    #if any(se_matrix <= 0):
    #    out = {'lr': None, 'p-value': None}
    #    return out

    #se = np.sqrt(se_matrix)

    # T-statistics for the b param of Weibull distribution
    #lr = (res.x[1] - 1) / (se[1])
    #p_value = 2 * min(1 - t.cdf(lr, len(tbf)), t.cdf(lr, len(tbf)))

    #print('444')
    
    lr = 2 * (cw_log_likelihood(res.x, tbf, censored=None, sign=1.0) - cw_log_likelihood((res.x[0], 1), tbf, censored=None, sign=1.0))

    p_value = 1 - chi2.cdf(lr, 1)

    out = {'lr': lr, 'p-value': p_value}
    #out
    #out = {'lr': lr, 'p-value': p_value}

    return out


def haas_dweibull(y: pd.DataFrame = None,
                  y_p: pd.DataFrame = None,
                  h_s: pd.DataFrame = None,
                  p: float = None,
                  **kwargs: dict) -> dict:
    """Haas Discrete Weibull test
    Parameters
    ----------
    y   : numerical pd.DataFrame with empirical P&L
    y_p : numerical pd.DataFrame with Value at Risk with level of p
    h_s : boolean pd.DataFrame with hit series (alternative to y < y_p)
    p   : numerical scalar of Value at Risk level
    Returns
    -------
    out : dictionary with test statistics and p-value
    Reference
    ---------
    Haas, M., 2005. Improved duration-based backtesting of value-at-risk. The Journal of Risk, 8(2), p.17.
    """

    if all(input_var is None for input_var in (y, y_p, h_s)):
        out = {'lr': None,
               'p-value': None}
        return out

    if h_s is None:
        h_s = (y < y_p)

    h_s = np.array(h_s, dtype=float)

    n = len(h_s)
    tbf = np.where(h_s)[0] + 1

    if len(tbf) < 1:
        # out = {'lr': None, 'p-value': None}
        # return out
        tbf = np.array(n + 1)

    tbf = np.diff(np.hstack((0, tbf)))

    censored = np.repeat(0, len(tbf)).tolist()
    if h_s[0] == 0:
        censored[0] = 1
    if h_s[-1] == 0:
        censored[-1] = 1

    a = -np.log(1 - p)
    res = minimize(dw_log_likelihood, np.array(10 ** -12), args=(a, tbf, censored, -1),
                   method='L-BFGS-B', options={'disp': False})

    # Terminate if solver couldn't find solution
    if not res.success or res.x < 10 ** (-10):
        out = {'lr': None, 'p-value': None}
        return out

    # Functional form of Discrete Weibull log-likelihood function
    dw_llik_fun = lambda b: dw_log_likelihood(b, a, tbf, censored=censored, sign=-1)

    # Calculate numerical hessian of dw_log_likelihood fun in the point res.x
    hess = finite_hessian(dw_llik_fun, res.x, h=10 ** (-6))

    # Check if hessian has been calculated
    if np.linalg.matrix_rank(hess) == 0:
        out = {'lr': None, 'p-value': None}
        return out

    # Calculate standard errors
    se_matrix = np.diag(np.linalg.inv(hess))
    if any(se_matrix <= 0):
        out = {'lr': None, 'p-value': None}
        return out
    se = np.sqrt(se_matrix)

    # T-statistics for the b param of Weibull distribution
    lr = (res.x - 1) / (se)
    p_value = 2 * min(1 - t.cdf(lr, len(tbf)), t.cdf(lr, len(tbf)))

    out = {'lr': lr[0], 'p-value': p_value[0]}

    return out


def engle_dq(y: pd.DataFrame = None,
             y_p: pd.DataFrame = None,
             h_s: pd.DataFrame = None,
             p: float = None,
             **kwargs: dict) -> dict:
    """Engle & Manganelli Dynamical Quantile
    Parameters
    ----------
    y   : numerical pd.DataFrame with empirical P&L
    y_p : numerical pd.DataFrame with Value at Risk with level of p
    h_s : boolean pd.DataFrame with hit series (alternative to y < y_p)
    p   : numerical scalar of Value at Risk level
    k   : numerical scalar of test's lag parameter
    Returns
    -------
    out : dictionary with test statistics and p-value
    Reference
    ---------
    Engle, R.F. and Manganelli, S., 2004. CAViaR: Conditional autoregressive value at risk by regression quantiles.
    Journal of Business & Economic Statistics, 22(4), pp.367-381.
    """

    if all(input_var is None for input_var in (y, y_p, h_s)):
        out = {'lr': None,
               'p-value': None}
        return out

    k = 5  # Set default value of lag parameter

    if 'params' in kwargs.keys():
        params = kwargs['params']
        if 'engle_dq' in params.keys():
            params = params['engle_dq']
            if 'k' in params.keys():
                k = int(params['k'])
    elif 'k' in kwargs.keys():
        k = int(kwargs['k'])

    if h_s is None:
        h_s = y < y_p

    hit_t = (h_s - p).values
    n = len(hit_t)

    # Create regression input (X and Y sets)
    x_set, y_set = [], []
    for i in range(k, n):
        x_set.append(np.append(1, hit_t[i - k:i].T))
        y_set.append(hit_t[i])
    x_set = np.array(x_set, dtype=float)
    y_set = np.array(y_set, dtype=float)

    # Check if not singular
    x_t_x = x_set.T @ x_set
    if x_set.shape[1] != np.linalg.matrix_rank(x_t_x):
        out = {'lr': None,
               'p-value': None}
        return out

    # Perform regression
    betas = np.linalg.inv(x_t_x) @ x_set.T @ y_set

    # Calculate likelihood and p-value
    lr = np.asscalar(betas.T @ x_set.T @ x_set @ betas / (p * (1 - p)))
    p_value = 1 - chi2.cdf(lr, k + 1)

    out = {'lr': lr, 'p-value': p_value}

    return out


def berkowitz_bl(y: pd.DataFrame = None,
                 y_p: pd.DataFrame = None,
                 h_s: pd.DataFrame = None,
                 p: float = None,
                 **kwargs: dict) -> dict:
    """ Berkowitz Box-Ljung statistic test
    Parameters
    ----------
    y   : numerical pd.DataFrame with empirical P&L
    y_p : numerical pd.DataFrame with Value at Risk with level of p
    h_s : boolean pd.DataFrame with hit series (alternative to y < y_p)
    p   : numerical scalar of Value at Risk level
    m   : numerical scalar of test specific parameter
    Returns
    -------
    out : dictionary with test statistics and p-value
    Reference
    ---------
    # Berkowitz, J., Christoffersen, P. and Pelletier, D., 2011. Evaluating value-at-risk models with desk-level data.
      Management Science, 57(12), pp.2213-2227.
    """

    if all(input_var is None for input_var in (y, y_p, h_s)):
        out = {'lr': None,
               'p-value': None}
        return out

    m = 5  # Set default value of autocorrelation lags

    # If user provided the specific parameter, then extract it from kwargs
    if 'params' in kwargs.keys():
        params = kwargs['params']
        if 'berkowitz_bl' in params.keys():
            params = params['berkowitz_bl']
            if 'm' in params.keys():
                m = int(params['m'])
    elif 'm' in kwargs.keys():
        m = int(kwargs['m'])

    if h_s is None:
        h_s = y < y_p

    n = int(len(h_s.values))
    i_p_seq = np.array(h_s - p).astype(float)

    lr = sum([auto_corr(i_p_seq, lag=k) ** 2 / (n - k) for k in range(1, m)])
    lr = n * (n + 2) * lr
    p_value = 1 - chi2.cdf(lr, m)

    out = {'lr': lr,
           'p-value': p_value}
    return out


def kramer_gini(y: pd.DataFrame = None,
                y_p: pd.DataFrame = None,
                h_s: pd.DataFrame = None,
                p: float = None,
                **kwargs: dict) -> dict:
    """ Kramer and Wied test based on Gini coefficient
    Parameters
    ----------
    y   : numerical pd.DataFrame with empirical P&L
    y_p : numerical pd.DataFrame with Value at Risk with level of p
    h_s : boolean pd.DataFrame with hit series (alternative to y < y_p)
    p   : numerical scalar of Value at Risk level
    Returns
    -------
    out : dictionary with test statistics and p-value
    Reference
    ---------
    # Krämer, W. and Wied, D., 2015. A simple and focused backtest of value at risk. Economics Letters, 137, pp.29-31.
    """

    if all(input_var is None for input_var in (y, y_p, h_s)):
        out = {'lr': None,
               'p-value': None}
        return out

    if h_s is None:
        h_s = y < y_p

    tbf = np.where(h_s)[0] + 1  # Assuming that time intex starts at 1
    # tbf = np.where(h_s)[0]

    if len(tbf) <= 1:
        out = {'lr': None,
               'p-value': None}
        return out

    tbf = np.diff(np.hstack((0, tbf)))  # Start measuring d_i from first point

    n = int(len(h_s.values))
    d = int(len(tbf))

    outer_substract = np.abs(np.subtract.outer(tbf, tbf))
    gini_coeff = sum(sum(outer_substract)) * (1 / d ** 2) / (2 * np.mean(tbf))

    lr = np.sqrt(n) * (gini_coeff - (1 - p) / (2 - p))

    p_value = 2 * min(1 - t.cdf(lr, d), t.cdf(lr, d))

    out = {'lr': lr,
           'p-value': p_value}
    return out
