# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 13:21:59 2023

@author: quent

Class to analyse thermal noise of any cantilever (raw data or extracted from an
.out file).
It performs a periodogram of the signal to find the resonnance frequency
and Q-factor.
It then infers k or S using the expression for the lorenzian one should find 
for a micro-mechanical oscillator.
"""

import seaborn as sns

custom_params = {
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.markeredgecolor": "k",
    "lines.markeredgewidth": 1.25,
    "figure.dpi": 200,
    "text.usetex": True,
    "font.family": "serif"}

sns.set_theme(context = "paper", 
              style="ticks", 
              font_scale=1, 
              rc=custom_params)

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import lmfit as lmf
import scipy.signal as sps

class TNoise:
    
    def __init__(self,data,sensi_measured,k_nom,
                 samp_freq=None, T=23, w_dir="cwd", old=False, high_speed = False):
        
        if w_dir != "cwd" :
            os.chdir(w_dir)
        
        if type(data) == str :
            with open(data,mode = 'r') as file:
                self.metadata = []
                n = -1
                while True :
                    line = file.readline()
                    if re.match("#",line):
                        self.metadata.append(line)
                        n += 1
                    else : 
                        break
                
                for line in self.metadata:
                    if re.match("# sampleFrequency:",line):
                        self.samp_freq = float(line.split()[-1])    
                    if re.match("# fancyNames:",line):
                        self.names = line.split("\"")[1::2]
                        self.where = {}
                        i = 0
                        for name in self.names :
                            self.where[name] = i
                            i+=1
                    if re.match("# units:",line):
                        self.units = line.split()[2:]
                
                if high_speed : 
                    detector = 'High Speed 1'
                else :
                    detector = 'Vertical Deflection' 
                
                def iter_func():
                    while True :
                        try :
                            line = file.readline()
                            if old :
                                temp_data = re.split(r' +',line)[:][self.where[detector]]
                            else :
                                temp_data = re.split(r' +',line)[1:][self.where[detector]]
                        except IndexError:
                            break
             		
                        yield temp_data
                
                self.vD = np.fromiter(iter_func(),dtype=float)
                
        elif samp_freq != None :
            self.vD=data
            self.samp_freq = samp_freq
          
        else :
            raise Exception("No sampling frequency provided")
        
        self.S_meas = sensi_measured
        self.k_nom = k_nom
        self.T = 273+T
        self.pow = 0
        
    def block_averaging(self, x, y, n_blocks):
        uniq = np.unique(np.log10(x)).astype(np.int32)
        umin, umax = uniq.min(), uniq.max() + 1
        blocks = np.logspace(umin,umax,np.int32(umax-umin)*n_blocks)
        blocks_y = np.zeros_like(blocks)
    
        blocks_ind = np.searchsorted(blocks, x, side='right')

        for i in range(len(blocks)) :
            mask = y[blocks_ind==i]
            if np.any(mask) :
                blocks_y[i] = np.mean(mask)
            else : 
                blocks_y[i] = np.nan
    
        nan_roi = ~np.isnan(blocks_y)
        blocks, blocks_y = blocks[nan_roi], blocks_y[nan_roi]
    
        return [blocks, blocks_y]
        
    def pow_spect(self, plot=False, w_welch=1e3, n_blocks = 100):

        if n_blocks == 0 :
            self.pow_freqs, self.pow = sps.welch(self.vD,
                                                 self.samp_freq,
                                                 scaling='density',
                                                 nperseg = w_welch)
        else :
             freqs, pow_spec = sps.welch(self.vD,
                                    self.samp_freq,
                                    scaling='density',
                                    nperseg = w_welch)
             self.pow_freqs, self.pow = self.block_averaging(freqs[1:], pow_spec[1:], n_blocks)
            
        if plot :
            plt.scatter(self.pow_freqs,self.pow,s=0.1)
            plt.xlabel(r'Frequencies ($\mathrm{Hz}$)')
            plt.ylabel(r'Power spectrum density ($\mathrm{V^2/Hz}$)')
            plt.xscale('log')
            plt.yscale('log')
        
        self.n_blocks = n_blocks
        self.w_welch = w_welch
                
    
    def TNaccurate(self, p0, nu, R, roi,
                   plot = False,
                   vary_S = False, vary_k = False, vary_m = True, vary_gamma = True, vary_C = False, vary_nu = False, vary_R = False,
                   weights = None, log = True):
        
        
                   
        def log_Flyv(f,gamma,m,k,S,C,nu,R):
            
            D = 1.38e-23*303 / gamma
            fv = nu / (np.pi*R**2)
            fc = k / (2*np.pi*gamma)
            fm = gamma / (2*np.pi*m*1e-11)
            
            res = D/(np.pi**2)
            res = res * ( 1 + np.sqrt(f/fv) )
            res = res / ( (fc - f**(3/2)/fv**(1/2) - f**2/fm )**2 + ( f + f**(3/2)/fv**(1/2) )**2 ) 
            
            return np.log(res/S**2 + C*1e-11)
        
        def lin_Flyv(f,gamma,m,k,S,C,nu,R):
            
            D = 1.38e-23*303 / gamma
            fv = nu / (np.pi*R**2)
            fc = k / (2*np.pi*gamma)
            fm = gamma / (2*np.pi*m*1e-11)
            
            res = D/(np.pi**2)
            res = res * ( 1 + np.sqrt(f/fv) )
            res = res / ( (fc - f**(3/2)/fv**(1/2) - f**2/fm )**2 + ( f + f**(3/2)/fv**(1/2) )**2 ) 
            
            return res/S**2 + C*1e-11
        
        if log :
            func = log_Flyv
            data_fit = np.log(self.pow[roi])
        else :
            func = lin_Flyv
            data_fit = self.pow[roi]
        
        Flyvbjerg_model = lmf.Model(func, independant_vars = ['f'])
        Flyvbjerg_param = Flyvbjerg_model.make_params(gamma = dict(value = p0[0],
                                                                   min = 0,
                                                                   vary = vary_gamma),
                                                      m = dict(value = p0[1]/1e-11,
                                                               min = 0,
                                                               vary = vary_m),
                                                      k = dict(value = p0[2],
                                                               min = 0,
                                                               vary = vary_k),
                                                      S = dict(value = p0[3],
                                                               min = 0,
                                                               vary = vary_S),
                                                      C = dict(value = p0[4]/1e-11,
                                                               min = 0,
                                                               vary = vary_C),
                                                      nu = dict(value = nu,
                                                                min = 0,
                                                                vary = vary_nu),
                                                      R = dict(value = R,
                                                               min = 0,
                                                               vary = vary_R))
        
        res = Flyvbjerg_model.fit(data_fit, Flyvbjerg_param, f = self.pow_freqs[roi])
        print(res.fit_report())
        
        if res.success :
            res.best_values['m'] *= 1e-11   
            res.best_values['C'] *= 1e-11     
            self.params = res.best_values
            if log :
                self.lorentz_fit = np.exp(res.best_fit)
                self.fit_uncert = [np.exp(res.best_fit - res.eval_uncertainty(sigma = 2)),
                                   np.exp(res.best_fit + res.eval_uncertainty(sigma = 2))]
            else :
                self.lorentz_fit = res.best_fit
                self.fit_uncert = [res.best_fit - res.eval_uncertainty(sigma = 2),
                                   res.best_fit + res.eval_uncertainty(sigma = 2)]
            try :
                res.uvars['m'] *= 1e-11
                res.uvars['C'] *= 1e-11
                self.uparams = res.uvars
            except :
                self.uparams = None
        
        if plot and res.success :
            plt.figure(figsize=(4,4))

            plt.plot(self.pow_freqs, self.pow, label='data')
            plt.fill_between(self.pow_freqs[roi],
                             self.fit_uncert[0],
                             self.fit_uncert[1],
                             color = "orange",
                             alpha=0.3,
                             zorder=999)
            plt.scatter(self.pow_freqs[roi], self.lorentz_fit,label='fit',c='darkorange',s=0.5,zorder=1000)    
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("Frequency ($\mathrm{Hz}$)")
            plt.ylabel("PSD ($\mathrm{V^2/Hz}$)")
            plt.legend()
            plt.tight_layout()


        
    def calib(self, f_0_est, 
              Q_est=None, p0 = None, roi = None,
              win=None, w_welch=0, f_spect=None, Q_spect=None,
              plot=True,
              vary_S=False,vary_Q=True,vary_f=True,vary_k=True,vary_C=True,
              corr_factor=1, weights=None, log=True,blocks_pdec=50):

        if type(self.pow) == int or w_welch != 0 :
            self.pow_spect(w_welch=w_welch, n_blocks = blocks_pdec)        
        
 
        if p0 != None :
            self.f_0_est = f_0_est
            self.Q_est = p0[0]
            self.k_nom = p0[1]
            self.S_meas = p0[2]
            self.C_est = p0[3]
        else :
            self.f_search = ((self.pow_freqs-f_0_est*1.5<0)*(self.pow_freqs-f_0_est*0.5>0))
            self.peaks = sps.find_peaks(self.pow[self.f_search],
                                        height = np.max(self.pow[self.f_search]))[0]
            try :
                self.peak_ind = self.peaks[0] + np.min(np.nonzero(self.f_search))
                self.f_0_est = self.pow_freqs[self.peak_ind]
                self.Q_est = self.f_0_est/(sps.peak_widths(self.pow[self.f_search], self.peaks)[0][0]*(self.pow_freqs[1]-self.pow_freqs[0]))
            except IndexError :
                self.f_0_est = f_0_est
                self.Q_est = 1
            if np.any(np.isinf(self.Q_est)) :
                print("Q could not be estimated automatically")
                if Q_est == None :
                    print("Please input an estimated value for Q using Q_est parameter.")
                    return
                self.Q_est = Q_est
        
        if f_spect is None :
            self.fmax = 1e10
            self.fmin = 1e-2
        else : 
            self.fmin = f_spect[0]
            self.fmax = f_spect[1]

        if Q_spect is None :
            self.Qmax = 1000
            self.Qmin = 0
        else : 
            self.Qmin = Q_spect[0]
            self.Qmax = Q_spect[1]
        
        def log_lorenz(f,f_0,Q,k_eff,S,C):
            res = 2*1.38e-23*self.T*f_0**3
            res = res / (np.pi*k_eff*Q)
            res = res / ((f**2-f_0**2)**2 + (f*f_0/Q)**2)
            res = res / S**2
            res = res * corr_factor
            return np.log(res + C)
        
        def lin_lorenz(f,f_0,Q,k_eff,S,C):
            res = 2*1.38e-23*self.T*f_0**3
            res = res / (np.pi*k_eff*Q)
            res = res / ((f**2-f_0**2)**2 + (f*f_0/Q)**2)
            res = res / S**2
            res = res * corr_factor
            return res + C
        
        if log :
            func = log_lorenz
            
            blocks_pdec
            data_fit = np.log(self.pow)
        else :
            func = lin_lorenz
            data_fit = self.pow
        
        lortz_model = lmf.Model(func, independant_vars = ['f'])
        lortz_param = lortz_model.make_params(f_0 = dict(value = self.f_0_est,
                                                     min = self.fmin,
                                                     max = self.fmax,
                                                     vary = vary_f),
                                            Q = dict(value = self.Q_est,
                                                     min = self.Qmin,
                                                     max = self.Qmax,
                                                     vary = vary_Q),
                                            k_eff = dict(value = self.k_nom,
                                                         min = self.k_nom*0.5,
                                                         max = self.k_nom*2,
                                                         vary = vary_k),
                                            S = dict(value = self.S_meas,
                                                     min = self.S_meas*0.5,
                                                     max = self.S_meas*2,
                                                     vary = vary_S),
                                            C = dict(value = self.C_est,
                                                     min=0,
                                                     max=1e-5,
                                                     vary=vary_C))
        
        
        
        if (win is None) and (roi is not None) :
            res = lortz_model.fit(data_fit[roi],
                                  lortz_param,
                                  f=self.pow_freqs[roi])
        elif (roi is None) and (win is None) :
            res = lortz_model.fit(data_fit[1:],
                                  lortz_param,
                                  f=self.pow_freqs[1:])
        elif (win is not None) and (roi is not None) :
            raise Exception("Both frequency window (win) and region of interest (roi) were filled. Chose between one of the two.")
        else:
            down, up = win[0], win[1]
            res = lortz_model.fit(data_fit[(self.pow_freqs-down>0)*(self.pow_freqs-up<0)],
                                  lortz_param,
                                  f=self.pow_freqs[(self.pow_freqs-down>0)*(self.pow_freqs-up<0)])
        
        print(res.fit_report())
        if res.success :    
            self.f_0 = res.best_values['f_0']
            self.Q = res.best_values['Q']
            self.k_eff = res.best_values['k_eff']
            self.S = res.best_values['S']
            self.C = res.best_values['C']
            self.R = res.rsquared
            if log :
                self.lorentz_fit = np.exp(res.best_fit)
                self.fit_uncert = [np.exp(res.best_fit - res.eval_uncertainty(sigma = 2)),
                                   np.exp(res.best_fit + res.eval_uncertainty(sigma = 2))]
            else :
                self.lorentz_fit = res.best_fit
                self.fit_uncert = [res.best_fit - res.eval_uncertainty(sigma = 2),
                                   res.best_fit + res.eval_uncertainty(sigma = 2)]
            try :
                self.uparams = res.uvars
            except :
                self.uparams = None
        
        if plot and res.success :
            plt.figure(figsize=(4,4))
            plt.plot(self.pow_freqs, self.pow,label='data')
            if (win is None) and (roi is not None) :
                plt.fill_between(self.pow_freqs[roi],
                                 self.fit_uncert[0],
                                 self.fit_uncert[1],
                                 color = "orange",
                                 alpha=0.3)

                plt.scatter(self.pow_freqs[roi], self.lorentz_fit,label='fit',c='darkorange',s=1,zorder=1000)    
       
            elif (roi is None) and (win is None) :
                plt.fill_between(self.pow_freqs[1:],
                                 self.fit_uncert[0],
                                 self.fit_uncert[1],
                                 color = "orange",
                                 alpha=0.3)
                plt.plot(self.pow_freqs[1:], self.lorentz_fit,label='fit',c='darkorange',zorder=1000)    
            
            else :
                plt.fill_between(self.pow_freqs[(self.pow_freqs-down>0)*(self.pow_freqs-up<0)],
                                 self.fit_uncert[0],
                                 self.fit_uncert[1],
                                 color = "orange",
                                 alpha=0.3)
                plt.plot(self.pow_freqs[(self.pow_freqs-down>0)*(self.pow_freqs-up<0)],
                         self.lorentz_fit,
                         label='fit',
                         c='darkorange',
                         zorder=1000)
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("Frequency ($\mathrm{Hz}$)")
            plt.ylabel("PSD ($\mathrm{V^2/Hz}$)")
            plt.legend()
            plt.tight_layout()
        
        
if __name__ == '__main__' :
    noise = TNoise("800kHz_10s.out",
                   15e-9,0.35)
