import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from pysabr import Hagan2002LognormalSABR
from pysabr import Hagan2002NormalSABR
import yfinance as yf

S0 = 100.0000000000001            # asset price at t = 0
T =1           # time in years
r = 0            # risk-free rate
N = 252*T              # number of time steps in simulation
M = 100             # number of simulations
# Heston dependent parameters
kappa = 1            # rate of mean reversion of variance under risk-neutral dynamics
theta = 0.20**2        # long-term mean of variance under risk-neutral dynamics
v0 = theta          # initial variance under risk-neutral dynamics
rho = -0.7              # correlation between returns and variances under risk-neutral dynamics
sigma = 0.5           # volatility of volatility
rng= np.random.default_rng(seed=1001)

# parameters for SABR
T = 1.0  # Time to maturity
r= 0 # interest rate
N= 252 # number of steps
alpha = 0.5 # Volatility of volatility
beta = 1   
p = -0.7   #Correlation 
v0= 0.2   # Initial volatility
S0 = 1  # Initial asset price
M=10000

symbol = "SPY" 
expiration_date = '2024-02-29'#'2023-12-15' # "2024-01-31" 
tte= 90/365 #15/365 # 61/365 # not in trading days, will change later


def main():
    #plotHistReturns()
    #F = ex10()
    #questionAB(F)
    #print('###')
    
    packSmile() # question C
    
    questionD()

def plotHistReturns(): 
    S,v = heston_model_sim(S0, v0, rho, kappa, theta, sigma,r,T, N, M)
    plt.plot(S)
    plt.show()
    plt.plot(v)
    plt.show()
    plt.plot(logRet)
    plt.show()
    logRet = [np.log(S[i]/S[i-1]) for i in range(1,len(S))]
    plt.hist(logRet, bins=100,histtype='step',density=True,linewidth=1)
    plt.xlabel('Log Returns')
    plt.ylabel('Frequency')
    plt.title('Histogram of Log Returns')

    plt.show()
    
def heston_model_sim(S0, v0, rho, kappa, theta, sigma,r,T, N, M):

    dt = T/N
    mu = np.array([0,0])
    cov = np.array([[1,rho],
                    [rho,1]])
    S = np.full(shape=(N+1,M), fill_value=S0)
    v = np.full(shape=(N+1,M), fill_value=v0)
    Z = np.random.multivariate_normal(mu, cov, (N,M))
    for i in range(1,N+1):
        S[i] = S[i-1] * np.exp( (r - 0.5*v[i-1])*dt + np.sqrt(v[i-1] * dt) * Z[i-1,:,0] )
        v[i] = np.maximum(v[i-1] + kappa*(theta-v[i-1])*dt + sigma*np.sqrt(v[i-1]*dt)*Z[i-1,:,1],0)
    
    return S, v

def sim_SABR(F0,v0, alpha, beta, p, r, T, N, M):
    # returns M simulated asset price and volatility time series of length N
    
    dt= T/N
    F, sigma= np.zeros((N+1, M)), np.zeros((N+1, M))
    F[0]=F0
    sigma[0]= v0
    
    covar= np.array([[1, p], [p,1]])
    dW= rng.multivariate_normal(np.zeros(2), covar, size=(N, M))
    
    for i in range(1,N+1):
        
        F[i]= F[i-1] + sigma[i-1] * F[i-1]**(beta) * dW[i-1,:,0]  * np.sqrt(dt)
        sigma[i]= sigma[i-1]* np.exp(-0.5* alpha**2 *dt + alpha * dW[i-1,:,1]* np.sqrt(dt))
        
    return F, sigma

def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T)* norm.cdf(d2)

def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return -S * norm.cdf(-d1) + K * np.exp(-r*T)* norm.cdf(-d2)


def mc_optionValue(F, K_values):
    
    call_values = pd.Series(data=np.zeros(len(K_values)), index=K_values)
    put_values = pd.Series(data=np.zeros(len(K_values)), index=K_values)
    
    for i in range(len(K_values)):
        
        call_payoffs = F -K_values[i]
        call_payoffs[call_payoffs<0]= 0

        put_payoffs = K_values[i]- F
        put_payoffs[put_payoffs<0]= 0
        
        call_values.iloc[i] = np.exp(-r*T) * np.mean(call_payoffs)
        put_values.iloc[i] = np.exp(-r*T) * np.mean(put_payoffs)
        
    return call_values, put_values
        

def ex10():
    
    F, sigma = sim_SABR(S0, v0, alpha, beta, p, r, T, N, M)
    
    plt.plot(F[:,0])
    plt.plot(sigma[:,0])
    plt.show()
    
    return F[-1,:]

def questionAB(F):
    
    K_values = np.array([0.75, 0.9, 1, 1.1, 1.25])

    call_valuesMC, put_valuesMC = mc_optionValue(F, K_values)
    
    print(call_valuesMC)
    
    # need to adjust variance, but not sure how
    bsCALL_values = np.array([BS_CALL(S0,K,T,r,v0) for K in K_values])
    bsPUT_values = np.array([BS_PUT(S0,K,T,r,v0) for K in K_values])
    
    plotMCBS(bsCALL_values, call_valuesMC, K_values, 'Calls')
    plotMCBS(bsPUT_values, put_valuesMC, K_values, 'Puts')
    
    

def plotMCBS(bs, mc,K, putORcall):
    
    fig, ax = plt.subplots()
    
    ax.scatter(K,bs, label='BS Value')
    ax.scatter(K, mc, label=' SABR Value')
    ax.set_title('SABR and Black-Scholes Comparision for' + putORcall, fontsize=20)
    ax.set_xlabel('Strike', fontsize=15)
    ax.set_ylabel('Value', fontsize=15)
    ax.legend()
    
    plt.show()
    

def packSmile():
    #tesing out package
    
    #K_values = np.array([0.75, 0.9, 1, 1.1, 1.25])
    K_values = np.linspace(0.7, 1.3, 100)
    
    #v_atm_n : volatility at the money. I think ideally we input IV from calculated SABR MC ATM price
    # doesnt seem to be any difference between lognormal and normal model
    sabr = Hagan2002LognormalSABR(f=S0, shift=r, t=T,v_atm_n=v0, beta=beta, rho=-0.7, volvol=alpha)
    
    '''
    I think the following are just two different methods to calculate black IV
    not sure which one is 'more appropriate' but lognorm results seem more sensible
    '''
    res1= [sabr.lognormal_vol(K) for K in K_values]
    res2= [sabr.normal_vol(K) for K in K_values]
    
    
    plt.plot(K_values, res1)
    plt.plot(K_values, res2)
    plt.title('Volatility Smile')
    plt.show()
    
def download_option_prices(symbol, expiration_date):
    # Download option chain data
    option_chain = yf.Ticker(symbol).option_chain(expiration_date)

    # Access call and put option data
    call_options = option_chain.calls
    put_options = option_chain.puts

    # Print the option data
    '''
    print("Call Options:")
    print(call_options)
    print("\nPut Options:")
    print(put_options)
    '''
    
    return call_options[['strike', 'impliedVolatility']]


def get_current_price(symbol):
    # doesnt work for some reason
    
    # Create a Ticker object for the given symbol
    ticker = yf.Ticker(symbol)

    # Get the current price
    current_price = ticker.info['regularMarketPrice']
    
    print(current_price)

    return current_price
    

def packFit(vols, S0):
    
    #data from 2024 31.Jan Calls SPY
    t= tte
    
    '''
    not sure how to pick beta put pretty sure 1 is best choice for stocks
    does not make big difference unless beta very low 
    '''
    sabr = Hagan2002LognormalSABR(S0, shift=r, t=t, beta=0.95)
    
    [alpha, rho, volvol] = sabr.fit(vols['strike'], vols['impliedVolatility']*100)
    print(alpha, rho, volvol)
      
def questionD():
    
    vols= download_option_prices(symbol, expiration_date)
    #lastPrice = get_current_price(symbol)
    
    packFit(vols, 458.87)
    
    '''
    big difference in results when using put or calls to value, maybe only supposed to use calls- documentation unclear
    different expirations also significantly different
    '''
    
    
if __name__ == '__main__':
    main()
