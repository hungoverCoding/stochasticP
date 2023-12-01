from pandas_datareader import data as pdr
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import scipy.stats as stats
import math
import seaborn as sns
from statistics import mean
from scipy.stats import norm
from scipy.stats import binom
from scipy.optimize import fsolve


S0_1 = 1000 #initial stock price
S0_2 = 700
sigma1 = 0.34
sigma2 = 0.51
rf= 0 
K=S0_1-S0_2 # strike price
T=1 # Time to maturity in years
M=10000 # number of monte carlo simulations
rng= np.random.default_rng(seed=1001)

def main():
    questionA()
    #questionB()
    portSigma = questionC()
    #print(portSigma)
    questionD(portSigma)
    questionE(portSigma)
    QuestionF(portSigma)
    
def questionA():
    print('Question A')
    print(payoff(M,0.3))

def questionB():
    plotRhovsPayoff2()
    S_1Paths, S_2Paths,log1,log2 = eulermethodGBM(0.3)
    plt.plot(S_1Paths)
    plt.plot(S_2Paths)
    plt.show()

def questionC():
    print('Question C')
    portSigma = calculateVarPortfolio()
    return portSigma

def questionD(portSigma):
    print('Question D')
    BSPrice = BS_CALL(K,K,T,rf,portSigma)
    print(BSPrice)
    
def questionE(portSigma):
    print('Question E')
    
    adjustedSigma = adjustSigma(portSigma, K)
    print(adjustedSigma)
    
    bachelierPrice = bachelier_option_price(K,K,T,rf,adjustedSigma)
    print(bachelierPrice)

def QuestionF(portSigma):
    
    K_values = np.linspace(0,1000, 100)
    #adjustedSigma = adjustSigma(portSigma, K_values)
    #print(adjustedSigma)
    
    bachelier_values = [bachelier_option_price(S0_1-S0_2,K,T,rf,adjustSigma(portSigma, K)) for K in K_values]
    #bs_values = [BS_CALL(S0_1-S0_2,K,T,rf,portSigma) for K in K_values]
    bs_values = [payoff(M, 0.3, K) for K in K_values]
    
    plt.plot(K_values, bachelier_values, marker='o', linestyle='-', color='b')
    plt.plot(K_values, bs_values, marker='o', linestyle='-', color='r')
    plt.show()

def calculateVarPortfolio():
    S1_list = [0]*M
    S2_list = [0]*M
    covar_list = [0]*M
    var1_list = [0]*M
    var2_list = [0]*M
    for i in range(M):
        a,b, S1_list[i],S2_list[i] = eulermethodGBM(0.3)
        covar_list[i] = np.cov(S1_list[i],S2_list[i])
        var1_list[i] = np.var(S1_list[i])
        var2_list[i] = np.var(S2_list[i])
    covar = np.mean(covar_list)*252
    var1 = np.mean(var1_list)*252
    var2 = np.mean(var2_list)*252
    portvar = 1**2*var1+(-1)**2* var2+2*1*(-1)*covar
    return np.sqrt(portvar)


def plotRhovsPayoff2():
    rho_values = np.linspace(-1,1, 100)
    Payoff2_values = [payoff(M,rho) for rho in rho_values]
    plt.plot(rho_values, Payoff2_values, marker='o', linestyle='-', color='b')
    plt.title('Call Prices for rho values')
    plt.xlabel('Rho')
    plt.ylabel('Call Price')
    plt.grid(True)
    plt.show()



def eulermethodGBM(rho):
    days = 1
    dt = days/252
    N = int(T/dt)
    S_1 = [0]*(N+1)
    S_1[0] = S0_1
    S_2 = [0]*(N+1)
    S_2[0] = S0_2
    eps1 = np.random.normal(loc=0, scale=1,size= N)
    eps2 = rho*eps1 + np.sqrt(1-rho**2)*np.random.normal(loc=0, scale=1,size= N)
    for t in range(1,N+1):
        S_1[t] = S_1[t-1]+S_1[t-1]*(rf*dt+sigma1*np.sqrt(dt)*eps1[t-1])
        S_2[t] = S_2[t-1]+S_2[t-1]*(rf*dt+sigma2*np.sqrt(dt)*eps2[t-1])
    S_1 = np.array(S_1)
    S_2 = np.array(S_2)
    log_returns1 = np.log(S_1[1:] / S_1[:-1])
    log_returns2 = np.log(S_2[1:] / S_2[:-1])
    return S_1,S_2,log_returns1,log_returns2

def Payoff2(M,rho):
    S1_list = [0]*M
    S2_list = [0]*M
    price_list = [0]*M
    eps1 = np.random.normal(loc=0, scale=1,size= M)
    eps2 = rho*eps1 + np.sqrt(1-rho**2)*np.random.normal(loc=0, scale=1,size= M)
    
    for i in range(M):
        
        S1_list[i] = S0_1*np.exp((rf-0.5*(sigma1)**2)*T+sigma1*np.sqrt(T)*eps1[i])
        S2_list[i] = S0_2*np.exp((rf-0.5*(sigma2)**2)*T+sigma2*np.sqrt(T)*eps2[i])
        price_list[i] = max(S1_list[i]-S2_list[i]-K,0)
    
    callValue = np.exp(-rf*T)*(sum(price_list)/M)
    
    return callValue

def payoff(M, rho, K= K):
    
    # generate errors
    covar=  np.array([[1, rho], [rho, 1]])
    eps = rng.multivariate_normal(np.zeros(2), covar, size=M).T
    
    # calculate stock prices and option payoffs
    S1 = S0_1*np.exp((rf-0.5*(sigma1)**2)*T+sigma1*np.sqrt(T)*eps[0])
    S2 = S0_2*np.exp((rf-0.5*(sigma2)**2)*T+sigma2*np.sqrt(T)*eps[1])
    spread = S1-S2
    payoffs = spread -K
    payoffs[payoffs<0]= 0
    
    callValue = np.exp(-rf*T)* np.sum(payoffs)/M
    
    '''
    plt.hist(spread,density=True, bins=30, edgecolor='black')
    plt.title('Histogram of Spreads')
    plt.ylabel('Density')
    plt.xlabel('Spread')
    plt.show()
    '''
    
    return callValue

def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T)* norm.cdf(d2)

def bachelier_option_price(S, K, T, r, sigma):
    
    d1 = (S - K) / (sigma * np.sqrt(T))
    option_price = np.exp(-r * T) * ((S - K) * norm.cdf(d1) + sigma * np.sqrt(T) * norm.pdf(d1))
    return option_price

def adjustSigma(sigma, mu):
    '''
    adjust Sigma used in geometric BM to sigma used for arithmetic BM
    '''
    
    return ((np.exp(sigma**2)-1)* np.exp(2*np.log(mu) +sigma**2))**0.5
    

def portVar(sigma1, sigma2, corr):
    # dont this is correct in its current form
    
    sigma1 = np.exp(2*sigma1**2) - np.exp(sigma1**2)
    sigma2 = np.exp(2*sigma2**2) - np.exp(sigma2**2)
    covar = corr* np.sqrt(sigma1) * np.sqrt(sigma2)
    
    
    var = sigma1 + sigma2 - 2* covar
    
    return np.sqrt(var)



if __name__ == '__main__':
    main()