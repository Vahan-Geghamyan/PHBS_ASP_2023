# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt
import pyfeng as pf

'''
MC model class for Beta=1
'''
class ModelBsmMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0, n_samples=1_000, n_steps=100):
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        Use self.bsm_model.impvol() method
        '''
        price = self.price(self, strike, spot)
        return self.bsm_model.impvol(price, strike, spot, texp)
    
    def price(self, strike, spot, texp=None, sigma=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
#         np.random.seed(12345)
        
        # Set T to expiration time
        T = texp
        delta_t = T / self.n_steps
        # Allocate space for storing the results
        prices = np.zeros((self.n_steps, self.n_samples))
        vols = np.zeros((self.n_steps, self.n_samples))
        
        # Generate Correlated Random Numbers
        W = np.random.randn(self.n_steps, self.n_samples)        
        Z = self.rho * W + np.sqrt(1 - self.rho ** 2) * np.random.randn(self.n_steps, self.n_samples)
        
        # Set the initial values
        prices[0, :] = spot
        vols[0, :] = self.sigma
        
        for k in range(self.n_steps-1):
            prices[k+1, :] = prices[k, :] * np.exp(vols[k, :] * W[k, :] * np.sqrt(delta_t) - 0.5 * vols[k, :]**2 * delta_t)
            vols[k+1, :] = vols[k, :] * np.exp(self.vov * Z[k, :] * np.sqrt(delta_t) - 0.5 * self.vov**2 * delta_t)
                
        return np.array([np.mean(cp*np.fmax(prices[-1, :] - K, 0)) for K in strike])
        

'''
MC model class for Beta=0
'''
class ModelNormalMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0, n_samples=1_000, n_steps=100):
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol.
        Use self.normal_model.impvol() method        
        '''
        price = self.price(self, strike, spot)
        return self.normal_model.impvol(price, strike, spot, texp)
        
    def price(self, strike, spot, texp=None, sigma=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
#         np.random.seed(12345)
        
        # Set T to expiration time
        T = texp
        delta_t = T / self.n_steps
        # Allocate space for storing the results
        prices = np.zeros((self.n_steps, self.n_samples))
        vols = np.zeros((self.n_steps, self.n_samples))
        
        # Generate Correlated Random Numbers
        W = np.random.randn(self.n_steps, self.n_samples)        
        Z = self.rho * W + np.sqrt(1 - self.rho ** 2) * np.random.randn(self.n_steps, self.n_samples)
        
        # Set the initial values
        prices[0, :] = spot
        vols[0, :] = self.sigma
        
        for k in range(self.n_steps-1):
            prices[k+1, :] = prices[k, :] + vols[k, :] * W[k, :] * np.sqrt(delta_t)
            vols[k+1, :] = vols[k, :] * np.exp(self.vov * Z[k, :] * np.sqrt(delta_t) - 0.5 * self.vov**2 * delta_t)
                
        return np.array([np.mean(cp*np.fmax(prices[-1, :] - K, 0)) for K in strike])

'''
Conditional MC model class for Beta=1
'''
class ModelBsmCondMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0, n_samples=1_000, n_steps=100):
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None):
        ''''
        should be same as bsm_vol method in ModelBsmMC (just copy & paste)
        '''
        price = self.price(self, strike, spot)
        return self.bsm_model.impvol(price, strike, spot, texp)
    
    def price(self, strike, spot, texp=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and BSM price.
        Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
#         np.random.seed(12345)
        
        # Set T to expiration time
        delta_t = texp / self.n_steps
        # Allocate space for storing the results
        prices = np.zeros(self.n_samples)
        option_prices = np.zeros((self.n_samples, len(strike)))
        vols = np.zeros((self.n_steps, self.n_samples))
        
        # Generate Random Numbers
        Z = np.random.randn(self.n_steps, self.n_samples)        
        
        # Set the initial values
        vols[0, :] = self.sigma
        
        for k in range(self.n_steps-1):
            vols[k+1, :] = vols[k, :] * np.exp(self.vov * Z[k, :] * np.sqrt(delta_t) - 0.5 * self.vov**2 * delta_t)
            
        sigma_T = vols[-1]
        I_T = np.mean(vols**2, axis=0)/self.sigma**2
        
        sigma_cmc = self.sigma * np.sqrt((1 - self.rho**2)* I_T)
        spot_cmc = spot * np.exp(self.rho/self.vov * (sigma_T - self.sigma) -\
                                 0.5 * self.rho**2 * self.sigma**2 * texp * I_T)
                   
        for i in range(self.n_samples):
            bsm_model = pf.Bsm(sigma_cmc[i], self.intr, self.divr, is_fwd=True)
#             option_prices[i, :] = np.array([bsm_model.price(K, spot_cmc[i], texp, cp) for K in strike])
            option_prices[i, :] = bsm_model.price(strike, spot_cmc[i], texp, cp)
            
        return np.mean(option_prices, axis=0)


'''
Conditional MC model class for Beta=0
'''
class ModelNormalCondMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0, n_samples=1_000, n_steps=100):
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp):
        ''''
        should be same as norm_vol method in ModelNormalMC (just copy & paste)
        '''
        price = self.price(self, strike, spot)
        return self.normal_model.impvol(price, strike, spot, texp)
        
    def price(self, strike, spot, texp, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
#         np.random.seed(12345)
        
        # Set T to expiration time
        delta_t = texp / self.n_steps
        # Allocate space for storing the results
        prices = np.zeros(self.n_samples)
        option_prices = np.zeros((self.n_samples, len(strike)))
        vols = np.zeros((self.n_steps, self.n_samples))
        
        # Generate Random Numbers
        Z = np.random.randn(self.n_steps, self.n_samples)        
        
        # Set the initial values
        vols[0, :] = self.sigma
        
        for k in range(self.n_steps-1):
            vols[k+1, :] = vols[k, :] * np.exp(self.vov * Z[k, :] * np.sqrt(delta_t) - 0.5 * self.vov**2 * delta_t)
            
        sigma_T = vols[-1]
        I_T = np.mean(vols**2, axis=0)/self.sigma**2
        
        sigma_cmc = self.sigma * np.sqrt((1 - self.rho**2)* I_T)
        spot_cmc = spot + self.rho/self.vov * (sigma_T - self.sigma)
                   
        for i in range(self.n_samples):
            normal_model = pf.Norm(sigma_cmc[i], self.intr, self.divr, is_fwd=True)
#             option_prices[i, :] = np.array([normal_model.price(K, spot_cmc[i], texp, cp) for K in strike])
            option_prices[i, :] = normal_model.price(strike, spot_cmc[i], texp, cp)
                
        return np.mean(option_prices, axis=0)
