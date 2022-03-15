# Codes were partially taken from coursera course "Investment Management with Python and Machine Learning Specialization"
# https://www.coursera.org/specializations/investment-management-python-machine-learning?
# partially from GitHub, https://github.com/GY400/Robust-Portfolio-Resampling/blob/master/Resampling_EF.ipynb
# partially created by the authors of the paper themselves

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Markowitz=================================================================================================================================

def compound(returns):
    """
    Returns the result of compounded return over the period of a set of returns
    """
    return np.expm1(np.log1p(returns).sum())

def annualized_returns(returns, periods_per_year):
    """
    Annualizes a set of returns
    We should include a number of periods per year
    """
    compounded_growth = np.prod(returns+1)
    n_periods = returns.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualized_cov(returns):
    """
    Returns an annualized covariance matrix, given set of returns (returns for exactly one year)
    """
    return returns.cov() * returns.shape[0]

def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights.T @ returns

def portfolio_vol(weights, cov):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    return np.sqrt(weights @ cov @ weights.T)

def sharpe(ret, vol, rf):
    return (ret-rf) / vol

def msr(riskfree_rate, er, cov):
    """
    Returns weights of a maximum Sharpe ratio portfolio
    """
    n = er.shape[0]                            
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n                   
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        Returns the negative of Sharpe ratio, given weights
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    results = minimize(neg_sharpe_ratio, init_guess, 
                       args = (riskfree_rate, er, cov), method = 'SLSQP',
                       options = {'disp': False},
                       constraints = (weights_sum_to_1),
                       bounds = bounds
                      )    
    return results.x

def gmv(cov):
    
    """
    Returns weights of a global minimum variance portfolio
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)

def minimize_vol(target_return, er, cov):
    """
    From target return to a weight vector
    """
    n = er.shape[0]                         
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n        
    return_is_target = {
        'type': 'eq',              
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    results = minimize(portfolio_vol, init_guess, 
                      args = (cov,), method = 'SLSQP',
                      options = {'disp': False},        
                      constraints = (return_is_target, weights_sum_to_1),
                      bounds = bounds
                      )    
    return results.x

def optimal_weights(n_points, er, cov):
    """
    Generates a list of weights to run the optimizer on 
    to minimize the volatility
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)   # Target returns, from min to max returns with a n-number of points
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def plot_ef(n_points, er, cov, show_msr=True, show_cml=True, show_ew=False, show_gmv=True, riskfree_rate=0):
    weights = optimal_weights(100, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    rets = pd.DataFrame(rets)
    vols = pd.DataFrame(vols)
    min_vol = vols.min()
    index = vols.loc[vols.isin([min_vol]).any(axis=1)].index.to_list()[0]
    rets = rets[index:]
    vols = vols[index:]
    ef = pd.concat([rets, vols], axis= 1).set_axis(['Efficient frontier', 'Volatility'], axis='columns')
    ax = ef.plot.line(x='Volatility', y='Efficient frontier', style='-')
    ax.set_xlim(left = 0, right = 0.50)
    ax.set_ylim(bottom = 0, top = 0.80)
    
    if show_msr:
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # Add MSR
        ax.plot([vol_msr], [r_msr], color='green', marker='o', markersize=8, label='MSR')
        
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # Add GMV
        ax.plot([vol_gmv], [r_gmv], color='red', marker='o', markersize=8, label='GMV')
    
    if show_cml:
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # Add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color = 'green', marker = 'o', linestyle = 'dashed', markersize=8, linewidth=2)
    
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # Add EW
        ax.plot([vol_ew], [r_ew], color='blue', marker='o', markersize=8, label='EW')
    
    ax.legend() 
    
    return ax

import matplotlib.pyplot as plt
import numpy as np
def plot_all(rf, er_1, cov_1, er_2, cov_2, title, name_1, name_2, name_3, name_4, name_5, name_6, name_7, name_8, ew=False, grid=False, crisis=False):
    """
    name_1: name of the first efficient frontier
    name_2: name of the second efficient frontier
    name_3: name of MSR portfolio on the first efficient frontier
    name_4: name of GMV portfolio on the first efficient frontier
    name_5: name of MSR portfolio on the second efficient frontier
    name_6: name of GMV portfolio on the secind efficient frontier
    name_7: name of realized MSR portfolio
    name_8: name of realized GMV portfolio
    """
    weights = optimal_weights(100, er_1, cov_1)
    rets = [portfolio_return(w, er_1) for w in weights]
    vols = [portfolio_vol(w, cov_1) for w in weights]
    rets = pd.DataFrame(rets)
    vols = pd.DataFrame(vols)
    min_vol = vols.min()
    index = vols.loc[vols.isin([min_vol]).any(axis=1)].index.to_list()[0]
    rets = rets[index:]
    vols = vols[index:]
    plt.plot(vols, rets, label=name_1, color='deepskyblue', linestyle=':')
    
    w_msr = msr(rf, er_1, cov_1)
    r_msr = portfolio_return(w_msr, er_1)
    vol_msr = portfolio_vol(w_msr, cov_1)
    plt.plot(vol_msr, r_msr, color='deeppink', marker='*', markersize=10, label=name_2)
    
    w_gmv = gmv(cov_1)
    r_gmv = portfolio_return(w_gmv, er_1)
    vol_gmv = portfolio_vol(w_gmv, cov_1)
    plt.plot(vol_gmv, r_gmv, color='lime', marker='v', markersize=8, label=name_3)
    
    weights_2 = optimal_weights(100, er_2, cov_2)
    rets_2 = [portfolio_return(w, er_2) for w in weights_2]
    vols_2 = [portfolio_vol(w, cov_2) for w in weights_2]
    rets_2 = pd.DataFrame(rets_2)
    vols_2 = pd.DataFrame(vols_2)
    min_vol_2 = vols_2.min()
    index = vols_2.loc[vols_2.isin([min_vol_2]).any(axis=1)].index.to_list()[0]
    rets_2 = rets_2[index:]
    vols_2 = vols_2[index:]
    plt.plot(vols_2, rets_2, label=name_4, color='dodgerblue')
    
    if crisis == True:
        ef_sd, r_rank, opt_weight = mv_ef(er_2, cov_2)
        msr_return, msr_vol, max_sharpe_ratio, msr_weights = msr_resampled(r_rank, ef_sd, opt_weight, rf)
        plt.plot(msr_vol, msr_return, color='r', marker='*', markersize=10, label=name_5)
    
    else:
        w_msr_2 = msr(rf, er_2, cov_2)
        r_msr_2 = portfolio_return(w_msr_2, er_2)
        vol_msr_2 = portfolio_vol(w_msr_2, cov_2)
        plt.plot(vol_msr_2, r_msr_2, color='red', marker='*', markersize=10, label=name_5)
    
    w_gmv_2 = gmv(cov_2)
    r_gmv_2 = portfolio_return(w_gmv_2, er_2)
    vol_gmv_2 = portfolio_vol(w_gmv_2, cov_2)
    plt.plot(vol_gmv_2, r_gmv_2, color='green', marker='v', markersize=8, label=name_6)
    
    port_return_msr = portfolio_return(w_msr, er_2)
    port_vol_msr = portfolio_vol(w_msr, cov_2)
    port_return_gmv = portfolio_return(w_gmv, er_2)
    port_vol_gmv = portfolio_vol(w_gmv, cov_2)
    plt.plot(port_vol_msr, port_return_msr, color='blue', marker='*', markersize=10, label=name_7)
    plt.plot(port_vol_gmv, port_return_gmv, color='purple', marker='^', markersize=8, label=name_8)
    
    if ew:
        n = er_2.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er_2)
        vol_ew = portfolio_vol(w_ew, cov_2)
        plt.plot(vol_ew, r_ew, color='darkorange', marker='o', markersize=8, label='In-sample EW')
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=3)
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    
    if grid:
        plt.grid(color='grey', linestyle='dashed', linewidth=0.3)
    
    plt.title(title)
    return plt.show()

def summary_expected(er, cov, riskfree_rate):
    weights_msr = msr(riskfree_rate, er, cov)
    r_msr = portfolio_return(weights_msr, er)
    vol_msr = portfolio_vol(weights_msr, cov)
    sharpe_msr = (r_msr - riskfree_rate)/vol_msr
    
    weights_gmv = gmv(cov)
    r_gmv = portfolio_return(weights_gmv, er)
    vol_gmv = portfolio_vol(weights_gmv, cov)
    sharpe_gmv = (r_gmv - riskfree_rate)/vol_gmv
    
    df1 = pd.DataFrame({
        'MSR reference': [r_msr.round(4), vol_msr.round(4), sharpe_msr.round(4), weights_msr.sum().round(4)],
        'GMV reference': [r_gmv.round(4), vol_gmv.round(4), sharpe_gmv.round(4), weights_gmv.sum().round(4)],
    }, index = ['Portfolio return','Portfolio volatility', 'Sharpe ratio', 'Sum of weights'])
    
    df2 = pd.DataFrame({
        'MSR reference': weights_msr.round(4),
        'GMV reference': weights_gmv.round(4),
    }, index = er.index + ' weight')
    
    return pd.concat([df1, df2], axis=0)

def summary_realized(er, cov, riskfree_rate, weights_msr, weights_gmv):
    r_msr = portfolio_return(weights_msr, er)
    vol_msr = portfolio_vol(weights_msr, cov)
    sharpe_msr = (r_msr - riskfree_rate)/vol_msr
    
    r_gmv = portfolio_return(weights_gmv, er)
    vol_gmv = portfolio_vol(weights_gmv, cov)
    sharpe_gmv = (r_gmv - riskfree_rate)/vol_gmv
    
    return pd.DataFrame({
        'MSR realized': [r_msr.round(4), vol_msr.round(4), sharpe_msr.round(4)],
        'GMV realized': [r_gmv.round(4), vol_gmv.round(4), sharpe_gmv.round(4)],
    }, index = ['Portfolio return','Portfolio volatility', 'Sharpe ratio'])

def comparison_exp_real(riskfree_rate, er_1, cov_1, er_2, cov_2):
    weights_msr = msr(riskfree_rate, er_1, cov_1)
    weights_gmv = gmv(cov_1)
    df1 = summary_expected(er_1, cov_1, riskfree_rate).iloc[:3]
    df2 = summary_realized(er_2, cov_2, riskfree_rate, weights_msr, weights_gmv)
    df2['MSR difference'] = df2['MSR realized'] - df1['MSR reference']
    df2['GMV difference'] = df2['GMV realized'] - df1['GMV reference']
    return pd.concat([df1, df2], axis=1)

def comparison_msr_gmv(riskfree_rate, er_1, cov_1, er_2, cov_2):
    df1 = summary_expected(er_1, cov_1, riskfree_rate)
    df2 = summary_expected(er_2, cov_2, riskfree_rate)
    df2['MSR difference'] = df2['MSR reference'] - df1['MSR reference']
    df2['GMV difference'] = df2['GMV reference'] - df1['GMV reference']
    df2 = df2.rename(columns = {'MSR reference': 'MSR current year', 'GMV reference': 'GMV current year'}, inplace = False)
    return pd.concat([df1, df2], axis=1)

def summary(er, cov, weights, riskfree_rate, msr=True):
    r = portfolio_return(weights, er.values)
    vol = portfolio_vol(weights, cov)
    sharpe = (r - riskfree_rate)/vol
    
    df1 = pd.DataFrame({
        'Portfolio': [r.round(4), vol.round(4), sharpe.round(4), weights.sum().round(4)],
    }, index = ['Portfolio return','Portfolio volatility', 'Sharpe ratio', 'Sum of weights'])
    
    df2 = pd.DataFrame({
        'Portfolio': weights.round(4),
    }, index = er.index + ' weight')
    
    df = pd.concat([df1, df2], axis=0)
    
    if msr is True:
        return df.set_axis(['MSR'], axis='columns')
    
    else:
        return df.set_axis(['GMV'], axis='columns')

def summary_full(er_1, cov_1, er_2, cov_2, weights_exp, weights_curr_year, riskfree_rate, msr=True):
    port_exp = summary(er_1, cov_1, weights_exp, riskfree_rate)
    port_curr_year = summary(er_2, cov_2, weights_curr_year, riskfree_rate)
    port_realized = summary(er_2, cov_2, weights_exp, riskfree_rate)
    
    diff_curr_exp = port_curr_year - port_exp
    diff_realized_exp = port_realized - port_exp
    diff_realized_curr = port_realized - port_curr_year
    
    df = pd.concat([port_exp, port_curr_year, port_realized, diff_curr_exp, diff_realized_exp, diff_realized_curr], axis= 1)
    
    if msr is True:
        columns_1 = ['MSR reference', 'MSR current year', 'MSR realized', 'MSR current year - MSR reference', 
                        'MSR realized - MSR reference', 'MSR realized - MSR current year']
        
        return df.set_axis(columns_1, axis='columns')
    
    else:
        columns_2 = ['GMV reference', 'GMV current year', 'GMV realized', 'GMV current year - GMV reference', 
                        'GMV realized - GMV reference', 'GMV realized - GMV current year']
        
        return df.set_axis(columns_2, axis='columns')
    
def mse(riskfree_rate, er_1, cov_1, er_2, cov_2):    
    realized_gmv = comparison_exp_real(riskfree_rate, er_1, cov_1, er_2, cov_2)['GMV realized'].iloc[0:3]   
    current_year_gmv = comparison_msr_gmv(riskfree_rate, er_1, cov_1, er_2, cov_2)['GMV current year'].iloc[0:3]
    tots2 = np.sqrt((realized_gmv-current_year_gmv)**2)
    
    realized_msr = comparison_exp_real(riskfree_rate, er_1, cov_1, er_2, cov_2)['MSR realized'].iloc[0:3]
    current_year_msr = comparison_msr_gmv(riskfree_rate, er_1, cov_1, er_2, cov_2)['MSR current year'].iloc[0:3]
    tots1 = np.sqrt((realized_msr-current_year_msr)**2) 
    
    return pd.DataFrame({
        'MSR': np.array(tots1),
        'GMV' : np.array(tots2),
    }, index=['Tracking error Return','Tracking error Volatility','Tracking error Sharpe ratio']).T

def squared_error(summary_full_msr, summary_full_gmv, year):    
    msr_ret_diff = (summary_full_msr['MSR realized - MSR current year'].iloc[0:1]**2).squeeze()
    msr_vol_diff = (summary_full_msr['MSR realized - MSR current year'].iloc[1:2]**2).squeeze()
    msr_sr_diff = (summary_full_msr['MSR realized - MSR current year'].iloc[2:3]**2).squeeze()
    
    gmv_ret_diff = (summary_full_gmv['GMV realized - GMV current year'].iloc[0:1]**2).squeeze()
    gmv_vol_diff =(summary_full_gmv['GMV realized - GMV current year'].iloc[1:2]**2).squeeze()
    gmv_sr_diff = (summary_full_gmv['GMV realized - GMV current year'].iloc[2:3]**2).squeeze()
        
    column_names = ['Squared error Return MSR', 'Squared error Volatility MSR', 'Squared error Sharpe ratio MSR',
                   'Squared error Return GMV', 'Squared error Volatility GMV', 'Squared error Sharpe ratio GMV']
  
    return pd.DataFrame({
        'A': msr_ret_diff,
        'B': msr_vol_diff,
        'C': msr_sr_diff,
        'D': gmv_ret_diff,
        'E': gmv_vol_diff,
        'F': gmv_sr_diff
    }, index=[year]).set_axis(column_names, axis=1)

def tracking_errors_total(squared_errors_mv, squared_errors_bl, squared_errors_res):
    mv = np.sqrt(squared_errors_mv.sum() / squared_errors_mv.shape[0])
    bl = np.sqrt(squared_errors_bl.sum() / squared_errors_bl.shape[0])
    res = np.sqrt(squared_errors_res.sum() / squared_errors_res.shape[0])
    df_1 = pd.DataFrame({'MSR': mv[:3].tolist(),
             'GMV': mv[3:].tolist()}, index=['Tracking error Return MVO', 
                                             'Tracking error Volatility MVO',
                                             'Tracking error Sharpe ratio MVO'])
    df_2 = pd.DataFrame({'MSR': bl[:3].tolist(),
             'GMV': bl[3:].tolist()}, index=['Tracking error Return BL', 
                                             'Tracking error Volatility BL',
                                             'Tracking error Sharpe ratio BL'])
    df_3 = pd.DataFrame({'MSR': res[:3].tolist(),
             'GMV': res[3:].tolist()}, index=['Tracking error Return Resampling', 
                                             'Tracking error Volatility Resampling',
                                             'Tracking error Sharpe ratio Resampling'])
    return pd.concat([df_1, df_2, df_3], axis=0)

def save_to_excel(path, df, name):
    """
    Creates a new excel file and adds all the DataFrames as new sheets to it
    Inputs:
    df: DataFrame or list or DataFrames
    name: name of a sheet ot list of names
    """
    writer = pd.ExcelWriter(path, engine='xlsxwriter')
    for i, j in zip(df, name):
        i.to_excel(writer, sheet_name = j)
    writer.save()
    writer.close()
    
def add_to_excel(path, df, name):
    """
    Adds DataFrames as new sheets to existing excel file
    Inputs:
    df: DataFrame or list or DataFrames
    name: name of a sheet ot list of names
    """
    from openpyxl import load_workbook
    writer = pd.ExcelWriter(path, engine='openpyxl', mode='a')
    for i, j in zip(df, name):
        i.to_excel(writer, sheet_name = j)
    writer.save()
    writer.close()
    

# Black-Litterman===========================================================================================================================    
    
def implied_excess_returns(delta, cov, weights):
    """
    Obtain the implied excess expected returns by reverse engineering the weights
    Inputs:
    delta: risk aversion coefficient (scalar)
    cov: variance-covariance matrix (N x N) as a DataFrame
    weights: portfolio weights (N x 1) as a Series
    Returns a N x 1 vector of returns as a Series
    Method .squeeze() is to get a Series from a 1-column DataFrame
    """
    weights = weights.squeeze()
    ir = delta * cov.dot(weights.values).squeeze()
    return ir

def p_matrix(cov, views, value=1):                 
    """
    Returns a P matrix for absolute views expressed for all the assets/stocks
    """
    p = pd.DataFrame(np.zeros([len(cov.columns), len(views.index)]), columns=cov.columns)
    np.fill_diagonal(p.values, value)
    return p

def proportional_omega(cov, tau, p):
    """
    Returns simplified omega
    Inputs:
    cov: N x N covariance matrix as a DataFrame
    tau: a scalar
    p: a K x N DataFrame linking views and assets
    Returns a P x P DataFrame, a Matrix representing prior uncertainties
    """
    omega = p.dot(tau * cov).dot(p.T)         
    diag = np.diag(omega)                      
    return pd.DataFrame(np.diag(diag)) 

def black_litterman(cov, weights, excess_views, P, omega=None, delta=2.5, tau=0.02):
    """
    Input parameters:
    N: a number of assets/stocks
    K: a number of active views
    cov: a N x N covariance matrix, a DataFrame
    weights: a N x 1 vector of weights, a Series or a 1-columns DataFrame
    views: a K x 1 vector of active views, a Series or a 1-columns DataFrame
    P: a K x N matrix linking views and assets/stocks, a DataFrame
    omega: a K x K matrix, a DataFrame, or None, if None, it is calculated proportionally to cov
    delta, tau: scalars
    Returns:
    posterior expected excess returns and covariance matrix
    """
    if omega is None:
        omega = proportional_omega(cov, tau, P)         
    if type(weights) == pd.DataFrame:
        weights = weights.squeeze()
    if type(excess_views) == pd.DataFrame:
        excess_views = excess_views.squeeze()                                
    ir = implied_excess_returns(delta, cov, weights)          
    cov_scaled = tau * cov                              
    first_term = cov_scaled.dot(P.T)                   
    second_term = P.dot(cov_scaled).dot(P.T) + omega    
    second_term_inv = np.linalg.inv(second_term)        
    third_term = excess_views.values - P.dot(ir)                                
    excess_mu_bl = ir + first_term.dot(second_term_inv).dot(third_term)
    forth_term = P.dot(cov_scaled)
    cov_bl = cov + cov_scaled - first_term.dot(second_term_inv).dot(forth_term)
    return excess_mu_bl, cov_bl

def bl_cov_no_views(cov, tau=0.02):
    return (1+tau)*cov

# Resampling===============================================================================================================================

import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

def mv_ef(mu, sigma, k=200):        
    mu = np.array(mu)
    sigma = np.array(sigma)
    n = len(mu)                    
    c = matrix(mu)                
    h = matrix(-np.zeros(n))       
    G = matrix(-np.identity(n))    
    b = matrix(1.0)                 
    A = matrix(np.ones(n)).T       

    r_min_solver = solvers.lp(c,G,h,A,b)       
    r_min_weight = np.array(r_min_solver['x'])                                       
    r_min = r_min_weight.T.dot(mu)                                                        
    r_max_solver = solvers.lp(-c,G,h,A,b)     
    r_max_weight = np.array(r_max_solver['x'])
    r_max = r_max_weight.T.dot(mu)            
    r_rank = np.linspace(r_min,r_max,k)        
    
    ef_sd = []
    opt_weight = []
    G_hat = -matrix(np.vstack((mu.T,np.identity(n))))   
                                                                     
    for r in r_rank:                                   
        P = 2 * matrix(sigma)
        q = matrix(np.zeros(n))                        
        zero = np.zeros(n)                             
        zero.shape = (n,1)                             
        h_hat = -matrix(np.vstack((np.array([r]),zero)))                          
        ef_solver = solvers.qp(P,q,G_hat,h_hat,A,b)      
        weight = np.array(ef_solver['x'])
        opt_weight.append(weight)        
        ef_sigma = np.sqrt(weight.T.dot(sigma.dot(weight))[0, 0])
        ef_sd.append(ef_sigma)          
        
    return ef_sd, r_rank, opt_weight                     
            
def r_ef(mu, sigma, k=200, m=500):                                                                 
    mu = np.array(mu)
    sigma = np.array(sigma)             
    size = len(mu)                                        
    REF_weights = [np.zeros((size,1)) for i in range(k)]  
                                                          
    REF_mean = []
    REF_sd = []
    for i in range(m):                                    
        sample = np.random.multivariate_normal(mu, sigma, size).T   
        sample_cov = np.cov(sample)
        sample_mean = np.mean(sample, 1)        
        ref_sd, ref_r, ref_opt_weights = mv_ef(sample_mean, sample_cov, k)
                                                
        for weight_REF, weight_ref in zip(REF_weights, ref_opt_weights):   
            weight_REF += weight_ref                
                                                
    for weight_REF in REF_weights:
        weight_REF /= m                          
        REF_mean.append(weight_REF.T.dot(mu)[0])                                      
        REF_sd.append(np.sqrt(weight_REF.T.dot(sigma.dot(weight_REF))[0,0]))   
        
    return REF_sd, REF_mean, REF_weights

def msr_resampled(REF_mean, REF_sd, REF_weights, rf):
    """
    Inputs:
    REF_mean: list of returns of portfolios on resampled efficient frontier, corresponding to each return point 
    gotten after applying the function r_ef, type = list
    REF_sd: list of volatilities of portfolios on resampled efficient frontier, corresponding to each return point 
    gotten after applying the function r_ef, type = list
    rf: risk-free rate, a float
    Returns:
    return of MSR portfolio return, a float
    volatility of MSR portfolio, a float
    sharpe ratio of MSR portfolio, a float
    weights of MSR portfolio, np.array
    """
    REF_mean = pd.DataFrame(REF_mean)
    REF_sd = pd.DataFrame(REF_sd)
    sharpe_ratio = ((REF_mean-rf) / REF_sd)
    max_sharpe_ratio = sharpe_ratio.max().squeeze()
    index = sharpe_ratio.loc[sharpe_ratio.isin([max_sharpe_ratio]).any(axis=1)].index.to_list()[0]
    msr_ref_port_return = REF_mean[index:index+1][0].squeeze()
    msr_ref_port_vol = REF_sd[index:index+1][0].squeeze()
    msr_ref_port_weights = REF_weights[index].squeeze()
    
    return msr_ref_port_return, msr_ref_port_vol, max_sharpe_ratio, msr_ref_port_weights

def gmv_resampled(REF_mean, REF_sd, REF_weights, rf):
    """
    Inputs:
    REF_mean: list of returns of portfolios on resampled efficient frontier, corresponding to each return point 
    specified in function r_ef, type = list
    REF_sd: list of volatilities of portfolios on resampled efficient frontier, corresponding to each return point 
    specified in function r_ef, type = list
    rf: risk-free rate, a float
    Returns:
    return of GMV portfolio, a float
    volatility of GMV portfolio, a float
    sharpe_ratio of GMV portfolio, a float
    weights of GMV portfolio, np.array
    """
    REF_mean = pd.DataFrame(REF_mean)
    REF_sd = pd.DataFrame(REF_sd)
    min_vol = REF_sd.min()
    index = REF_sd.loc[REF_sd.isin([min_vol]).any(axis=1)].index.to_list()[0]
    gmv_ref_port_return = REF_mean[index:index+1].squeeze()
    gmv_ref_port_vol = REF_sd[index:index+1].squeeze()
    gmv_sharpe_ratio = (gmv_ref_port_return-rf) / gmv_ref_port_vol
    gmv_ref_port_weights = REF_weights[index].squeeze()
    
    return gmv_ref_port_return, gmv_ref_port_vol, gmv_sharpe_ratio, gmv_ref_port_weights

def plot_ref(REF_mean, REF_sd, REF_weights, ann_returns, ann_cov, show_msr=True, show_cml=True, show_ew=False, show_gmv=True, rf=0):

    REF_mean = pd.DataFrame(REF_mean)
    REF_sd = pd.DataFrame(REF_sd)
    ref = pd.concat([REF_mean, REF_sd], axis= 1).set_axis(['Efficient frontier', 'Volatility'], axis='columns')
    ax = ref.plot.line(x='Volatility', y='Efficient frontier', style='-')
    ax.set_xlim(left = 0, right = 0.50)
    ax.set_ylim(bottom = 0, top = 0.80)
    
    if show_msr:
        msr_ref_port_return, msr_ref_port_vol, max_sharpe_ratio, msr_ref_port_weights = msr_resampled(REF_mean, REF_sd, REF_weights, rf)
        # Add MSR
        ax.plot([msr_ref_port_vol], [msr_ref_port_return], color='green', marker='o', markersize=8, label='MSR')
        
    if show_gmv:
        gmv_ref_port_return, gmv_ref_port_vol, gmv_sharpe_ratio, gmv_ref_port_weights = gmv_resampled(REF_mean, REF_sd, REF_weights, rf)
        # Add GMV
        ax.plot([gmv_ref_port_vol], [gmv_ref_port_return], color='red', marker='o', markersize=8, label='GMV')
    
    if show_cml:
        msr_ref_port_return, msr_ref_port_vol, max_sharpe_ratio, msr_ref_port_weights = msr_resampled(REF_mean, REF_sd, REF_weights, rf)
        # Add CML
        cml_x = [0, msr_ref_port_vol]
        cml_y = [rf, msr_ref_port_return]
        ax.plot(cml_x, cml_y, color = 'green', marker = 'o', 
                linestyle = 'dashed', markersize=8, linewidth=2)
        
    if show_ew:
        n = ann_returns.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, ann_returns)
        vol_ew = portfolio_vol(w_ew, ann_cov)
        # Add EW
        ax.plot([vol_ew], [r_ew], color='blue', marker='o', markersize=8, label='EW')

    ax.legend() 
    
    return ax

def plot_ref_all(REF_mean, REF_sd, REF_weights, ann_returns, ann_cov, rf, title, name_1, name_2, name_3, name_4, name_5, name_6, name_7, name_8, show_msr_gmv=False, ew=False, grid=False, crisis=False):
    """
    name_1: name of the first efficient frontier
    name_2: name of MSR portfolio on the first efficient frontier
    name_3: name of GMV portfolio on the first efficient frontier
    name_4: name of the second efficient frontier
    name_5: name of MSR portfolio on the second efficient frontier
    name_6: name of GMV portfolio on the secind efficient frontier
    name_7: name of realized MSR portfolio
    name_8: name of realized GMV portfolio
    """
    # Markowitz efficient frontier
    weights = optimal_weights(100, ann_returns, ann_cov)
    rets = [portfolio_return(w, ann_returns) for w in weights]
    vols = [portfolio_vol(w, ann_cov) for w in weights]
    rets = pd.DataFrame(rets)
    vols = pd.DataFrame(vols)
    min_vol = vols.min()
    index = vols.loc[vols.isin([min_vol]).any(axis=1)].index.to_list()[0]
    rets = rets[index:]
    vols = vols[index:]
    plt.plot(vols, rets, label=name_1, color='deepskyblue')
    
    # MSR portfolio on Markowitz efficient frontier
    if crisis:
        ef_sd, r_rank, opt_weight = mv_ef(ann_returns, ann_cov)
        msr_return, msr_vol, max_sharpe_ratio, msr_weights = msr_resampled(r_rank, ef_sd, opt_weight, rf)
        plt.plot(msr_vol, msr_return, color='r', marker='*', markersize=10, label=name_2)
    
    else:
        w_msr = msr(rf, ann_returns, ann_cov)
        r_msr = portfolio_return(w_msr, ann_returns)
        vol_msr = portfolio_vol(w_msr, ann_cov)
        plt.plot(vol_msr, r_msr, color='red', marker='*', markersize=10, label=name_2)
        
    # GMV portfolio on Markowitz efficient frontier
    w_gmv = gmv(ann_cov)
    r_gmv = portfolio_return(w_gmv, ann_returns)
    vol_gmv = portfolio_vol(w_gmv, ann_cov)
    plt.plot(vol_gmv, r_gmv, color='green', marker='v', markersize=8, label=name_3)
    
    # Resampled efficient frontier
    plt.plot(REF_sd, REF_mean, label=name_4, color='hotpink')
    # MSR portfolio on resampled efficient frontier
    msr_ref_port_return, msr_ref_port_vol, max_sharpe_ratio, msr_ref_port_weights = msr_resampled(REF_mean, REF_sd, REF_weights, rf)
    plt.plot(msr_ref_port_vol, msr_ref_port_return, color='deeppink', marker='*', markersize=10, label=name_5)
    # GMV portfolio on resampled efficient frontier
    gmv_ref_port_return, gmv_ref_port_vol, gmv_sharpe_ratio, gmv_ref_port_weights = gmv_resampled(REF_mean, REF_sd, REF_weights, rf)
    plt.plot(gmv_ref_port_vol, gmv_ref_port_return, color='lime', marker='v', markersize=8, label=name_6)
    
    
    # Realised MSR and GMV portfolios
    if show_msr_gmv:
        port_return_msr = portfolio_return(msr_ref_port_weights, ann_returns)
        port_vol_msr = portfolio_vol(msr_ref_port_weights, ann_cov)
        port_return_gmv = portfolio_return(gmv_ref_port_weights, ann_returns)
        port_vol_gmv = portfolio_vol(gmv_ref_port_weights, ann_cov)
        plt.plot(port_vol_msr, port_return_msr, color='blue', marker='*', markersize=10, label=name_7)
        plt.plot(port_vol_gmv, port_return_gmv, color='purple', marker='^', markersize=8, label=name_8)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=3)
        
    else:
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=2)
        
    if ew:
        n = ann_returns.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, ann_returns)
        vol_ew = portfolio_vol(w_ew, ann_cov)
        plt.plot(vol_ew, r_ew, color='darkorange', marker='o', markersize=8, label='In-sample EW')
    
    if grid:
        plt.grid(color='grey', linestyle='dashed', linewidth=0.3)
        
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.title(title)
    
    return plt.show()
    
def summary_ref(REF_sd, REF_mean, REF_weights, ann_returns, rf, msr=True):
    if msr is True:
        msr_ref_port_return, msr_ref_port_vol, max_sharpe_ratio, msr_ref_port_weights = msr_resampled(REF_mean, REF_sd, REF_weights, rf)
        sum_to_one_1 = msr_ref_port_weights.sum()
        df1 = pd.DataFrame({
        'MSR': [msr_ref_port_return.round(4), msr_ref_port_vol.round(4), max_sharpe_ratio.round(4), msr_ref_port_weights.sum().round(4)],
        }, index = ['Portfolio return','Portfolio volatility', 'Sharpe ratio', 'Sum of weights'])
        df2 = pd.DataFrame({
        'MSR': msr_ref_port_weights.round(4),
        }, index = ann_returns.index + ' weight')
        
        return pd.concat([df1, df2], axis=0).set_axis(['MSR'], axis='columns')
    
    else:
        gmv_ref_port_return, gmv_ref_port_vol, gmv_sharpe_ratio, gmv_ref_port_weights = gmv_resampled(REF_mean, REF_sd, REF_weights, rf)
        sum_to_one_2 = gmv_ref_port_weights.sum().round(4)
        df1 = pd.DataFrame({
        'GMV': [gmv_ref_port_return.round(4), gmv_ref_port_vol.round(4), gmv_sharpe_ratio.round(4), gmv_ref_port_weights.sum().round(4)],
        }, index = ['Portfolio return','Portfolio volatility', 'Sharpe ratio', 'Sum of weights'])
        df2 = pd.DataFrame({
        'GMV': gmv_ref_port_weights.round(4),
        }, index = ann_returns.index + ' weight')
        
        return pd.concat([df1, df2], axis=0).set_axis(['GMV'], axis='columns')

def summary_ref_full(REF_sd, REF_mean, REF_weights, ann_returns, ann_cov, rf, p_msr=True):  # In this finction p_msr, in all the other simply msr
    if p_msr is True:
        msr_ref_port_return, msr_ref_port_vol, max_sharpe_ratio, msr_ref_port_weights = msr_resampled(REF_mean, REF_sd, REF_weights, rf)
        port_exp = summary_ref(REF_sd, REF_mean, REF_weights, ann_returns, rf, msr=p_msr)
        weights_curr_year = msr(rf, ann_returns, ann_cov)
        port_curr_year = summary(ann_returns, ann_cov, weights_curr_year, rf, msr=p_msr)
        port_realized = summary(ann_returns, ann_cov, msr_ref_port_weights, rf, msr=p_msr)
        
        diff_curr_exp = port_curr_year - port_exp
        diff_realized_exp = port_realized - port_exp
        diff_realized_curr = port_realized - port_curr_year
        
        df_1 = pd.concat([port_exp, port_curr_year, port_realized, diff_curr_exp, diff_realized_exp, diff_realized_curr], axis= 1)
        columns_1 = ['MSR reference', 'MSR current year', 'MSR realized', 'MSR current year - MSR reference', 
                        'MSR realized - MSR reference', 'MSR realized - MSR current year']
        
        return df_1.set_axis(columns_1, axis='columns')
    
    else:
        gmv_ref_port_return, gmv_ref_port_vol, gmv_sharpe_ratio, gmv_ref_port_weights = gmv_resampled(REF_mean, REF_sd, REF_weights, rf)
        port_exp = summary_ref(REF_sd, REF_mean, REF_weights, ann_returns, rf, msr=p_msr)
        weights_curr_year = gmv(ann_cov)
        port_curr_year = summary(ann_returns, ann_cov, weights_curr_year, rf, msr=p_msr)
        port_realized = summary(ann_returns, ann_cov, gmv_ref_port_weights, rf, msr=p_msr)
        
        diff_curr_exp = port_curr_year - port_exp
        diff_realized_exp = port_realized - port_exp
        diff_realized_curr = port_realized - port_curr_year
        
        df_2 = pd.concat([port_exp, port_curr_year, port_realized, diff_curr_exp, diff_realized_exp, diff_realized_curr], axis= 1)
        columns_2 = ['GMV reference', 'GMV current year', 'GMV realized', 'GMV current year - GMV reference', 
                        'GMV realized - GMV reference', 'GMV realized - GMV current year']    
        
        return df_2.set_axis(columns_2, axis='columns')