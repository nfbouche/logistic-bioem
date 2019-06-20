# coding=utf-8

from astropy.table import Table,Column
import re , sys
import numpy as np

from matplotlib import pyplot as p
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats,special

import logging
import pandas as pd
import pymc3
import theano
import theano.tensor as t
import sys
if sys.version_info.major<3.:
    import cPickle as pickle # python 2
else:
    import pickle
    
SEED = 12345 #for reproducibility


p.rcParams['font.size']=18
p.rcParams['lines.linewidth']=3.0
p.rcParams['axes.linewidth']=3.0
p.rcParams['image.interpolation']='none'
p.rcParams['xtick.labelsize']='large'
p.rcParams['ytick.labelsize']='large'
p.rcParams['xtick.minor.visible']=True
p.rcParams['ytick.minor.visible']=True
p.rcParams['xtick.minor.size']=4.0
p.rcParams['xtick.major.size']=6.0
p.rcParams['ytick.minor.size']=4.0
p.rcParams['ytick.major.size']=6.0
p.rcParams['legend.numpoints']=1
p.rcParams['image.origin']='lower'
p.rcParams['font.family']='serif'
p.rcParams['font.serif']='Computer Modern'

def findall(pat,text):
	match=re.findall(pat,text)
	#match=re.findall(pat,text)
	if match:
		#cleans from '-'
		return match
	else:
		return None

def trace_quantiles(x):
    return pd.DataFrame(pymc3.quantiles(x, [2.5, 16, 50., 84, 97.5]) )

    
def read_data(filename='Hamalgue'):

    if '.tex' in filename:
        tab = Table.read(filename,format='latex',data_start=3)
        tab.rename_column('B-field','Bfield(muT)')
        c = tab['Melatonin level']
        n = np.zeros(len(c))
        for i,o in enumerate(c):
            if 'some' in o :
                n[i]=0.5
            if 'Not' in o:
                n[i]=0
            if 'Changed'==o:
                n[i]=1
        col=Column(data=n)
        tab.add_column(col,name='Changes')
    elif '.csv' in filename:
        tab = Table.read(filename, format='csv')
    else:
        tab = Table.read(filename,format='ascii.commented_header',guess=False,delimiter='|')
 
    #allow for both 'duration' and 'Duration' in table
    if 'duration' in tab.keys():
        exposure='duration'
    elif 'Duration' in tab.keys():
        exposure='Duration'
        
    #float values
    floats=np.array([float(tab[exposure][i][:-1]) for i,s in enumerate(tab[exposure])])

    #flag
    hours = [findall('h',s)!=None for s in tab[exposure]]
    days  = [findall('d',s)!=None for s in tab[exposure]]
    months= [findall('m',s)!=None for s in tab[exposure]]
    weeks = [findall('w',s)!=None for s in tab[exposure]]
    years = [findall('y',s)!=None for s in tab[exposure]]

    hours_float = hours* floats
    days_hours  = days * floats*24
    months_hours= months* floats*24*31
    weeks_hours = weeks * floats*24*7
    year_hours  = years * floats*24*365

    float_hours = hours_float + days_hours + months_hours + weeks_hours + year_hours
    col = Column(data=float_hours,name='hours')
    tab.add_column(col)

    #dealing with masked 
    col=tab['Bfield(muT)']
    col.fill_value =  -99 #Warning
    #print col.dtype
    if col.dtype.type is np.string_:
        ma=np.ma.array(data=col.data,mask=col.data=='--',fill_value=-99)
        a =np.where(ma.mask==False,ma.data,None)
        ma=np.ma.array(data=a.astype('float32'),mask=col.data=='--')
        tab['Bfield(muT)']=ma    
    else:
        tab['Bfield(muT)']=col
    
    
    return tab
  
def plot_kde(chain_name,names,axs,col,labels,lw=4):

        with open(chain_name+'.pkl', 'rb') as buff:
            data = pickle.load(buff)  

        basic_model, trace = data['model'], data['trace']    
        for i,var in enumerate(names):
            #axs[i]=pymc3.kdeplot(trace[var],ax=axs[i],color=col,lw=lw)
            axs[i]=pymc3.kdeplot(trace[var],ax=axs[i],color=col,lw=lw,label=labels)
        
        return axs

def compare_chains(chain1='humansAll_model1', chain2='ratsAll_model1', outname=None, fig=None):
        """
        chain1, chain2 = str  
        """
        
        if outname is not None:
            fig1=p.figure(5,figsize=(4,12))
            p.clf()
            ax1=fig1.add_subplot(211)
            ax2=fig1.add_subplot(212)
        
        elif fig is not None:
            ax1,ax2=fig
            
        #p.hist(trace['alpha'],normed=True,bins=50)
        ax1.set_xlim([-1,4])
        ax1.set_xlabel(r'$\alpha$')
        ax2.set_xlabel(r'$\beta$')
        ax2.set_xlim([-1,3]) 

       

        axs1=plot_kde(chain1,['alpha','beta'],[ax1,ax2],'b','Humans')
        ax1.set_title(r'$P(\theta)$ model 1')
         
        
        if chain2 is not None:
            if 'All' in chain2:
                label2='rats All'
            elif 'Bgt50' in chain2:
                label2='rats $|B|>=50\mu$T'
            elif 'Blt50' in chain2:
                label2='rats $|B|<50\mu$T'
                
            axs=plot_kde(chain2,['alpha','beta'],[ax1,ax2],'r',label2,lw=2)
            ax1.legend(loc=1)

        #ig1.subplots_adjust(hspace=0.7)
              
        if outname is not None:
            p.savefig(outname)
            
        return None

######################################################## 
    
class Histogram_Bfield:

    def __init__(self, humans, rats, verbose=True):
        if verbose:
            print("Reading from %s" % (humans) )
        tab_H = read_data(humans)
        tab_H.filled(-99)
        
        if verbose:
            print("Reading from %s" % (rats) )
        tab_rats = read_data(rats)
        
        self.tab_H=tab_H
        self.tab_rats = tab_rats
        
    def make_plot(self, outpdf=None, binsize=0.5, verbose=False):
        binsize=0.5
        p.figure(1,figsize=(10,9))
        p.clf()        
        h1,_,_ = p.hist(np.log10(self.tab_rats['Bfield(muT)']),bins=np.r_[-2+binsize/2:3.5:binsize],label='Rat studies',alpha=0.5)
        h0,_,_ = p.hist(np.log10(self.tab_H['Bfield(muT)']),bins=np.r_[-2+binsize/2:3.5:binsize],label='Human studies',alpha=0.75,hatch='/')
        p.axvline(np.log10(50),label='Earth $B_{\odot}\simeq$%2d$\mu$T' % (50),color='k',ls=':')
        p.xlabel('$\log B[\mu$T]')
        p.legend()
        tt,Pval= stats.ks_2samp(self.tab_rats['Bfield(muT)'],self.tab_H['Bfield(muT)'])
        if verbose:
            print("Pval KS: ",Pval,tt)
        self.Pval = Pval
        if outpdf is not None:
            if verbose:
                print("Saving to "+outpdf)
            p.savefig(outpdf)
    

#############################################
# 
# Logistic regression
#
#############################################  
    
class Model:
    """
    filename : str
        filename of data.tex or data.dat
    Nsample : int [default=10000]
        number of iterations per chain
    burnin : int [default=1000]
        number of burnins
    outname : str
        rootname for output
    shared : boolean [default True]
    threshold:  [default None]
        if not None split run in [a,b] range
    """

    logger = logging.getLogger('Model:')

    def __init__(self, filename, Nsample=10000, burnin=1000, outname=None, shared=True, threshold=None, with_outliers=None):
        self.logger.info("reading data %s " % (filename) )

        self.filename=filename
        self.outname=outname
        print("Reading from %s" %(filename) )
        self.data = read_data(filename)
        self.init_model(threshold)
        self.with_outliers=with_outliers
        
        if Nsample>1:
            self.run_model(Nsample=Nsample,burnin=burnin, shared=shared)
            self.get_summary()
            self.print_intervals()
            try:
                self.Watanabe(outname=outname)#incl Deviance
            except:
                self.logger.warning("Failed to save %s" %(outname) )
                
            self.save_run(outname=outname)
        else:
            self.read_run(outname=outname)
            self.get_summary()
            self.print_intervals()
            try:
                self.read_Watanabe(fname=outname)#incl Deviance
            except:
                self.logger.warning("Failed to read %s" %(outname) )
                
        #save bics
         
        return None
        
    def init_model(self, threshold):
        
        ##warning
        
        table = self.data.filled()## 
        #table.sort('Bfield(muT)')
        if threshold is not None:
            a,b=threshold
            self.Bthreshold = b
            condition = (table['Bfield(muT)']>=a) * (table['Bfield(muT)']<b)  #selecting in ange [a,b]
            sub = table[condition]
            table=sub.copy()
    
        #
        days = np.array(table['hours']/24.)
        log_days = np.log10(days)
        col = table['Bfield(muT)']
        bfield_log = np.log10(col.data)
        outcomes = np.array(table['Changes'])

         
        if isinstance(col,np.ma.masked_array):
            # https://github.com/pymc-devs/pymc3/issues/1254
            is_obscured = col.mask==True
            bfield_log_obscured = np.log10(col[is_obscured])
            n_obscured = bfield_log_obscured.shape[0]
            bfield_log_observed = np.log10(col[~is_obscured] )

            #print bfield_log.fill_value,is_obscured.sum()
            flag = bfield_log<4 # bfield_log<1.7
            #print(len(flag))
            log_days=log_days[flag]
            outcomes=outcomes[flag]
            bfield_log=bfield_log.data[flag]
            #print bfield_log,outcomes,log_days

        self.x_shared = theano.shared(log_days)
        self.bfield_shared = theano.shared(bfield_log)
    
        self.outcomes=outcomes
        self.bfield_log=bfield_log
        self.log_days=log_days
        
        self.N=len(self.outcomes)
    
     
        print("LEN (Outcomes) %d" % (len(outcomes)))

    def logistic(self, x):
            return t.exp(x) / (1 + t.exp(x))
    
    def construct_model(self, prob):
    
        with self.mymodel:
            
            step = None
            
            if self.with_outliers == None:

                o = pymc3.Bernoulli('outcome', prob, observed=self.outcomes) #likelihood
                  
            elif self.with_outliers=='Mixture':
                ##########With Mixture
                #pi = pymc3.Dirichlet('pi',a=np.ones(2))  #returns pi, 1-pi from Beta distribution
                pi = pymc3.Uniform('f_out',0,0.5)# Outliers are minority
                #pi = pymc3.Beta('f_out',1.,1.)
                
                BoundedNormal = pymc3.Bound(pymc3.Normal, lower=0, upper=1)
                cst = BoundedNormal('p_out',mu=0.5,sd=1.)#prior
                
                outliers = pymc3.Bernoulli.dist(p=cst)
                inliers  = pymc3.Bernoulli.dist(p=prob)

                o = pymc3.Mixture('outcome', w=[1-pi,pi], comp_dists=[inliers, outliers], observed=self.outcomes )#as before check?
    

            elif self.with_outliers=='Mixture_05':
                ##########With Mixture
                pi = pymc3.Uniform('f_out',0,0.5)# Outliers are minority
                #pi = pymc3.Dirichlet('pi',a=np.ones(2))  #returns pi, 1-pi from Beta distribution
                #pi = pymc3.Beta('f_out',1.,1.)
                
                #BoundedNormal = pymc3.Bound(pymc3.Normal, lower=0, upper=1)
                #cst = BoundedNormal('p_out',mu=0.5,sd=1.)#prior
                cst = pymc3.floatX(0.5) 
                
                outliers = pymc3.Bernoulli.dist(p=cst)
                inliers  = pymc3.Bernoulli.dist(p=prob)

                o = pymc3.Mixture('outcome', w=[1-pi,pi], comp_dists=[inliers, outliers], observed=self.outcomes )#as before check?

            elif self.with_outliers=='Robust_LR05':
            
                ###########From book "Bayesian analysis with python" O. Martin
                pi = pymc3.Uniform('f_out',0,0.5)# Outliers are minority
                #pi = pymc3.Beta('f_out',1.,1.) #Uniform equivalent
                p = 0.5 * pi + (1-pi) * prob  #outliers have p=0.5 
                
                o = pymc3.Bernoulli('outcome',p=p,observed=self.outcomes) #
                
            elif self.with_outliers=='Robust_LR':
            
                ###########From book
                pi = pymc3.Uniform('f_out',0,0.5)# Outliers are minority
                #pi = pymc3.Beta('f_out',1.,1.) #Uniform equivalent
                #p = 0.5 * pi + (1-pi) * prob  #outliers have p=0.5 
                BoundedNormal = pymc3.Bound(pymc3.Normal, lower=0, upper=1)
                cst = BoundedNormal('p_out',mu=0.5,sd=1.)#prior
                p = cst * pi + (1-pi) * prob  #outliers have p=0.5 
                
                o = pymc3.Bernoulli('outcome',p=p,observed=self.outcomes) #
            
            elif self.with_outliers=='Experimental':
            
                
                ## Define Bernoulli inlier / outlier flags according to a hyperprior
                #pi = pymc3.Uniform('f_out',0,0.5)# Outliers are minority
                frac_outliers = pymc3.Uniform('f_out', lower=0.0, upper=0.5)
                is_outlier = pymc3.Bernoulli('is_outlier', p=frac_outliers, shape=self.N)
                #
                #is_outlier = pymc3.Beta('is_outlier', 0.5, 0.5, shape=self.N)
                #
                #from Hogg's example https://docs.pymc.io/notebooks/GLM-robust-with-outlier-detection.html
                #http://bebi103.caltech.edu.s3-website-us-east-1.amazonaws.com/2017/tutorials/t6a_outliers.html
                # Set up normal distributions that give us the logp for both distributions
                #inliers  = pm.Normal.dist(mu=yest_in, sd=sigma_y_in).logp(yobs)
                #outliers = pm.Normal.dist(mu=yest_out, sd=sigma_y_in + sigma_y_out).logp(yobs)
                # Build custom likelihood, a potential will just be added to the logp and can thus function
                # like a likelihood that we would add with the observed kwarg.
                #o = pymc3.Potential('outcome', ((1 - is_outlier) * inliers).sum() + (is_outlier * outliers).sum())
                #issue with above: does not have observed variable
                
                yobs = theano.shared(np.asarray(self.outcomes, dtype=theano.config.floatX))
                            
                
                
                def loglike(data):
                    #similarly:
                    #BoundedNormal = pymc3.Bound(pymc3.Normal, lower=0, upper=1)
                    #cst = BoundedNormal('p_out',mu=0.5,sd=1.)#prior
                    cst=pymc3.floatX(0.5)
                    
                    log_like_bad  =  pymc3.Bernoulli.dist(p=cst).logp(data)
                    log_like_good =  pymc3.Bernoulli.dist(p=prob).logp(data)
                    
                    #return ((1 - is_outlier) * inliers)  + (is_outlier * outliers) 
                    return t.log( (1 - is_outlier) * t.exp(log_like_good) + is_outlier * t.exp(log_like_bad))
                    
                o = pymc3.DensityDist('outcome', loglike ,observed=yobs)

                #mixing steps not allowed with NUTS
                #step = pymc3.step_methods.Metropolis() #[alpha,  beta, frac_outliers, is_outlier])
                #step = [ pymc3.step_methods.Metropolis(self.mymodel.deterministics), pymc3.step_methods.BinaryMetropolis(['is_outlier']) #[alpha,  beta, cst, f_out, is_outlier])
                
                                                             
            elif self.with_outliers=='Categorical':
                
                ###########with Categorical
                #pi = pymc3.Beta('f_out',1.,1.)
                #frac_outliers = pymc3.Uniform('f_out', lower=0.0, upper=0.5)
                #is_outlier = pymc3.Bernoulli('is_outlier', p=frac_outliers, shape=self.N)
                
                is_outlier = pymc3.Beta('is_outlier', 0.5, 0.5, shape=self.N) #too slow
                
                #
                step = pymc3.step_methods.Metropolis()
                
                #BoundedNormal = pymc3.Bound(pymc3.Normal, lower=0, upper=1)
                #cst = BoundedNormal('p_out',mu=0.5,sd=1.)#prior
                cst=pymc3.floatX(0.5)
                
                #
                
                probs = t.switch(is_outlier<0.5,prob,cst)
                
                o=pymc3.Bernoulli('outcome', p=probs , observed=self.outcomes )
            
 
            else:
                raise "Model not supported\n [None, Robust_LR05, Robust_LR, Mixture_05, Mixture, Experimental, Categorical]"         

        return step
    
    def read_run(self,outname=None):
        
        with open(outname+'.pkl', 'rb') as buff:
            data = pickle.load(buff)  

        self.mymodel, self.trace = data['model'], data['trace']  

        self.trace_data = pymc3.trace_to_dataframe(self.trace)
        
        self.outname=outname
         
        return None
        
    def save_run(self,outname=None):
    
        with self.mymodel:
            if outname is not None: #saveing chain
                with open(outname+'.pkl', 'wb') as buff:
                    pickle.dump({'model': self.mymodel, 'trace': self.trace}, buff)
        
        self.outname=outname        
                
        return None
        
    def get_summary(self):
        #print pymc3.summary(self.trace)
        
        self.varnames = pymc3.utils.get_default_varnames(self.trace.varnames,include_transformed=False)
        #compute mean and medians                        
        self.summary = pymc3.summary(self.trace,self.trace.varnames,stat_funcs=[lambda x:pd.Series(np.mean(x, 0), name='mean'),trace_quantiles])
        #print(summary['mean'])
        print(self.summary)
        
        self.k=len(self.varnames)

        return self.summary
    
    def best_params_(self):
        
        summary = pymc3.summary(self.trace,self.trace.varnames,stat_funcs=[lambda x:pd.Series(np.mean(x, 0), name='mean'),trace_quantiles])
        #best parameters
        best_p = summary[50.0].to_dict() 
        
        return best_p 

    def best_params(self):
        
        summary = pymc3.summary(self.trace,self.trace.varnames,stat_funcs=[lambda x:pd.Series(np.mean(x, 0), name='mean'),trace_quantiles])
        #best parameters
        best_p = summary[50.0].to_dict()
        logp_best = self.mymodel.logp(best_p)
    
        
        return best_p, logp_best

    def print_intervals(self, ci=0.95, threshold=0.68):
        """
        ci confidence level for parameter range
        threshold for being an outliers 
        """
        my_range={}
        for k in self.varnames: #self.trace_data.keys():
            interval = np.percentile(self.trace[k],[(1-ci)*100./2,50.,100-(1-ci)*100./2])
            my_range[k] = "%.1f [%.1f-%.1f] (%2d\\%%) " % (interval[1],interval[0],interval[2],ci*100) 
            print(k,my_range[k])
            if 'beta' in k:   
                my_range['days']= "%.1f [%.1f-%.1f] (%2d\\%%)" % (10**interval[1],10**interval[0],10**interval[2],ci*100)
                print(k+' (days)',my_range['days'])


        if self.with_outliers == 'Categorical':
            is_outlier = np.array([self.trace_data['is_outlier__%d'%(i)].mean() for i in range(self.N)])>threshold
            fraction_outlier = is_outlier.sum()/np.float(self.N)
            self.is_outlier = is_outlier
            my_range['f_out'] = "%.2f ($P>$%.2f)" % (fraction_outlier, threshold)
            print('f_out: ',my_range['f_out'])
             
        self.my_range = my_range

        return None
       
        
    def Watanabe(self, outname=None):
        
        k=len(self.varnames)
        N=len(self.outcomes)  
       
        #Information content
        waic = pymc3.waic(self.trace,self.mymodel ) #{mymodel:trace},ic='WAIC') #or LOO Leave one out.
        print("\nWAIC %.2f +/- %.2f peff %.2f" % (waic.WAIC,waic.WAIC_se,waic.p_WAIC))
        self.WAIC = waic.WAIC
        
        waic_deviance = waic.WAIC - 2* waic.p_WAIC
        print("WAIC deviance",waic_deviance)
        LOO = pymc3.loo(self.trace,self.mymodel ) #{mymodel:trace},ic='WAIC') #or LOO Leave one out.
        print("LOO %.2f +/- %.2f peff %.2f" % (LOO.LOO,LOO.LOO_se,LOO.p_LOO))
        self.LOO=LOO.LOO

        
        logp_fct = self.mymodel.logp #log probability density function
        #lnp = np.array([model.logp(trace.point(i,chain=c)) for c in trace.chains for i in range(len(trace))]) #slow!
        self.lnp = np.array([logp_fct(self.trace.point(i,chain=c)) for c in self.trace.chains for i in range(len(self.trace))]) #fast
    
        #the elementwise log-posterior for the sampled trace.
        logp_post = pymc3.stats._log_post_trace(self.trace,model=self.mymodel) 
        ln_post = special.logsumexp(logp_post,axis=1)

        
        self.WBIC= N * (np.power(np.mean(-self.lnp),1./np.log(N))) /2. #why factor of 2
        print("WBIC ",self.WBIC   )
        
        #add Deviance IC
        try:
            DICm, DICp = self.Deviance()
            self.DIC=DICp #will be using this one
        except:
            self.DIC= -99
            
        arr = np.array([self.WAIC, self.LOO, self.WBIC, self.DIC])
        tab = Table(arr,names=['WAIC','LOO','WBIC','DIC'])
        tab.write(sys.stdout,format='ascii.fixed_width')   
        
        
        
        
        if outname is not None:
            tabname = outname+'_wbics.txt'
            tab.write(tabname,format='ascii.fixed_width', overwrite=True) 
    
        return tab
        
    def read_Watanabe(self, fname=None):

        tab = None
        if fname is not None:
            tabname = fname+'_wbics.txt'
            
            if 'txt' in tabname:
                tab = Table.read(tabname, format='ascii.fixed_width')
                tab.write(sys.stdout,format='ascii.fixed_width', overwrite=True) 
            
            self.WAIC = tab['WAIC']
            self.LOO = tab['LOO']
            self.WBIC= tab['WBIC']
            self.DIC = tab['DIC']    
        return tab

    def Deviance(self):
    
        k=len(self.varnames)
        N=len(self.outcomes)  
       
       
        #The BIC is an asymptotic result derived under the assumptions that the data distribution is in an exponential family
        #https://discourse.pymc.io/t/evaluate-logposterior-at-sample-points/235
        
        #likelihood at best p
        best_p, logp_best = self.best_params()
        
        logp_fct = self.mymodel.logp #log probability density function
        #lnp = np.array([model.logp(trace.point(i,chain=c)) for c in trace.chains for i in range(len(trace))]) #slow!
        self.lnp = np.array([logp_fct(self.trace.point(i,chain=c)) for c in self.trace.chains for i in range(len(self.trace))]) #fast
    
    
        #likelihood at Lmax
        idx = np.argmax(self.lnp)
        logp_max=self.lnp[idx]
        
        

        ###########################
        #Deviance
        Deviance_m = -2. * logp_max
        print('Deviance Lmax %.4f' % ( Deviance_m) )  # https://en.wikipedia.org/wiki/Deviance_information_criterion
        ############################
        
        ############################
        #DIC
        P = 2 * (logp_max-np.mean(self.lnp))
        DICm = -2 * (logp_max - P )
        print("DIC Lmax %.4f %.2f" % (DICm, P) )
        ###########################

        ###########################
        #Deviance
        Deviance_p = -2. * logp_best
        print('Deviance best %.4f' % (Deviance_p) ) # https://en.wikipedia.org/wiki/Deviance_information_criterion
        ############################
        
        ############################
        #DIC
        P = 2 * (logp_best-np.mean(self.lnp))
        DICp = -2 * (logp_best - P )
        print("DIC best %.4f %.2f" % ( DICp, P) )
        ###########################

       

        return DICm, DICp
        
    def BICs(self, mode='best'):
        """compute DIC, BIC etc 
        mode = 'best' | 'Lmax' 
        at the best/max likelihood
        """
        
        #likelihood at best p
        best_p, logp_best = self.best_params()
        
        logp_fct = self.mymodel.logp #log probability density function
        #lnp = np.array([model.logp(trace.point(i,chain=c)) for c in trace.chains for i in range(len(trace))]) #slow!
        self.lnp = np.array([logp_fct(self.trace.point(i,chain=c)) for c in self.trace.chains for i in range(len(self.trace))]) #fast
    
        #likelihood at Lmax
        idx = np.argmax(self.lnp)
        logp_max=self.lnp[idx]
        
        if mode=='best':
            L = logp_best
        elif mode=='Lmax':
            L = logp_max
        else:
            raise Error
        
        ############################
        #BIC
        #BIC = ln(n) k - 2 ln(L)        # https://en.wikipedia.org/wiki/Bayesian_information_criterion
        BIC = lambda logp : self.k * np.log(self.N)  - 2.0 * logp         
        print("BIC (%s)",BIC(L), self.N, self.k)
        ############################
        
        
        #############################
        #AIC
        # https://en.wikipedia.org/wiki/Akaike_information_criterion
        AIC = lambda logp: (- 2. * logp + 2. * self.k ) #/ N   
        AICc= lambda logp: (-2. *logp + 2. * self.k + 2. * self.k * (self.k+1.) / (self.N-self.k-1.) ) # Lu 2011 MathGeosc
        print("AIC (%s)",AIC(L), self.N, self.k)
        print("AICc (%s)",AICc(L), self.N, self.k)
        ############################
                
        return BIC(L),AIC(L),AICc(L)
    
         
    def plot_wtime(self, title='Human studies', ci=0.95, outname=None, text=True, ycust=None, ):
    
        if outname is not None:
            p.figure(1,figsize=(10,8))
            p.clf()
        #p.title('Human studies')
        #p.title(title)
        p.axvline(0,linestyle='-.',color='k')
        p.text(np.log10(1.2),1.2,'Day')
        p.axvline(np.log10(31),linestyle='-.',color='k')
        p.text(np.log10(31*1.2),1.2,'Month')
        p.axvline(np.log10(365),linestyle='-.',color='k')
        p.text(np.log10(365*1.2),1.2,'Year')
        p.axhline(0,linestyle=':',color='k',linewidth=4,xmin=0.01,xmax=1e5,alpha=0.4)
        p.axhline(1,linestyle=':',color='k',linewidth=4,xmin=0.01,xmax=1e5,alpha=0.4)
        #p.scatter(10**self.log_days,self.outcomes,marker='o',s=150,alpha=0.75,lw=2,edgecolor='k',label=title)
        if 'Human' in title:
            p.scatter(self.log_days,self.outcomes,marker='s',s=100,alpha=0.75,lw=1,edgecolor='k',label=title)        
        elif 'Rat' in title:
            p.scatter(self.log_days,self.outcomes,marker='o',s=100,alpha=0.75,lw=1,edgecolor='k',label=title)        
        
        #if self.with_outliers:
        #    pass
        #p.xscale('log')
        #p.semilogx(float_hours/24,tab['Changes'],'o',ms=10,mec='k',lw=2,label='Data')
        ##p.semilogx(float_hours[-2:],tab['Changes'][-2:],'o',mfc=None,ms=10,color='grey')
        if ycust is not None:
            p.ylim(ycust)
        else:
            p.ylim([-0.35,1.5])
        p.xlim([-1.8,3.5])
        p.ylabel('no Effect / Effect')
        p.yticks([0,1])
        p.xlabel(r'$\log$ $T$ [d]')
        if text:
            p.text(-1.6,0.05,'No effect on MLT')
            p.text(-1.6,1.05,'Effect on MLT')
        
        ###ONLY for model1:
        if isinstance(self,model1):
            params = self.get_summary()[50.0].to_dict()
            q = np.linspace(-2,4)
            pbest=self.logistic(params['alpha'] * (q - params['beta']) ).eval()
            #pbest=F(q,A[1],B[1])
            mp=1.-pbest
            n=self.N
            z=stats.norm(0,1).isf((1-ci)/2.)
            
            #confidence interval Pearson
            # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
            plow3 = (pbest+z**2/2./n)/(1+z**2/n) - z/(1+z**2/n) * np.sqrt(pbest*mp/n + z**2/4./n**2)
            phigh3= (pbest+z**2/2./n)/(1+z**2/n) + z/(1+z**2/n) * np.sqrt(pbest*mp/n + z**2/4./n**2)
             
            #adding to plot 1
            p.plot(q,pbest, label='Best model',color='r')
            ax=p.gca()
            #p.plot(10**q,plow3,'k-')
            #p.plot(10**q,phigh3,'k-')
            ax.fill_between(q,plow3,phigh3,color='grey',alpha=0.4)
        p.legend(loc=3)
     
        if outname is not None:
            p.savefig(outname)
     
    def sample_posterior(self, X, num_ppc_samples=2000):
        #Posterior predictive checks
        #on linear grid
        self.x_shared.set_value( X )
        
        with self.mymodel:
           ppc = pymc3.sample_ppc(self.trace, samples=num_ppc_samples)
          
        return ppc
   
    def predict(self, X, num_ppc_samples=10000, with_errors=False, ci=0.95):
        #xx,yy=np.meshgrid(np.arange(-1.8,3.51,0.1),np.arange(-2,3.51,0.1))
        
        #params = self.get_summary()[50.0].to_dict()
        
        ppc = self.sample_posterior(X, num_ppc_samples=num_ppc_samples)
        
        if with_errors==False:
            return ppc['outcome'].mean(axis=0)
        else:
            return ppc['outcome'].mean(axis=0), ['outcome'].std(axis=0)
        
    def plot_trace(self):
        pymc3.traceplot(self.trace,combined=False)
        
   
    def plot_corner(self, Nsig=3):
        try:
           import corner
        except:
            raise "corner package required"
                     
        #corner.corner(trace)
        r=np.ones(len(self.trace_data.keys()))*0.97
        perc = np.percentile(self.trace_data,(16,84),axis=(0,))#1sigma
        m = np.median(self.trace_data,axis=0)
        s = perc[1,:] - perc[0,:]
        corner.corner(self.trace_data,range=zip(m-Nsig*s,m+Nsig*s),truths=m, vars=self.trace_data.keys() )

        
class model1(Model):

    
    def run_model(self, Nsample, burnin,   shared=True):
        if shared:
            xdata = self.x_shared
        else:
            xdata = self.log_days
        
            
        with pymc3.Model() as self.mymodel:
            #alpha = pymc3.Normal('alpha', mu=0, sd=10) #not bound
            BoundedNormal = pymc3.Bound(pymc3.Normal, lower=-5, upper=5)
            alpha = BoundedNormal('alpha', mu=0, sd=10)
            beta = pymc3.Uniform('beta',lower=-2, upper=3.5)

            # 
            prob = self.logistic(alpha * (xdata - beta))             # P = C * (logT - B)
    
                      
            step = self.construct_model(prob)
            
            
            init = pymc3.find_MAP()

            self.burn=burnin
            if step is None:
                self.trace = pymc3.sample(Nsample,chains=2, n_init=burnin, start=init,  \
                     nuts_kwargs={'target_accept':0.8,'integrator':'two-stage'}, random_seed=SEED)
            else:
                self.trace = pymc3.sample(Nsample,chains=2, n_init=burnin, start=init, step=step, random_seed=SEED)
                    
            #save to pandas   
            self.trace_data = pymc3.trace_to_dataframe(self.trace)

    def predict_true(self,  X, params=None):
        
        #compute model
        if params is None:
            params = self.best_params_()
            
        prob = self.logistic(params['alpha'] * (X - params['beta']))
        
        return prob.eval()
    
    def plot_model2D(self, markersize=100, outname=None, edc='None', add_random=None, ax=None, add_colorbar=True, title=r"Parametric LR model"):
    
        params = self.get_summary().to_dict()
        #print(params)
        #plt.plot()
        xx,yy=np.meshgrid(np.arange(-1.8,3.51,0.1),np.arange(-2,3.51,0.1) )
        Z=self.predict_true(xx)
        if ax is None:
            ax=p.gca()
            
        im=ax.pcolormesh(xx,yy,Z,cmap=p.cm.Greys,alpha=0.8, edgecolors=edc)
        #p.pcolor(xx,yy,Z,cmap=p.cm.Greys,alpha=0.5)
        #p.contourf(xx,yy,Z,cmap=p.cm.Greys,alpha=0.5)
        
        beta = r"$\beta$=%.1f$^{+%.1f}_{%.1f}$" %(params[50.0]['beta'],params[84.0]['beta']-params[50.0]['beta'],params[16.0]['beta']-params[50.0]['beta'])
        ax.axvline(params[50.0]['beta'],ls='-',color='k', label=beta)
        #plt.contour(xx,yy,Z,[0.5])

        if add_random is None:
            x,y = self.log_days,self.bfield_log
        else:
            x,y = self.log_days+add_random[:len(self.outcomes)],self.bfield_log
        
     
        if add_colorbar:
            #axins1 = inset_axes(ax,
            #            width="50%",  # width = 10% of parent_bbox width
            #            height="5%",  # height : 50%
            #            loc='lower right')
            cax=ax.figure.add_axes([0.1,0.92,0.2,0.03])
            cb=ax.figure.colorbar(im, cax=cax,  orientation='horizontal')
            cb.set_clim([0,1])
            cb.set_ticks([0.2,0.5,0.8])
        
        ax.scatter(x,y,marker='s',c=self.outcomes*2,label='Human studies',cmap=p.cm.Paired,edgecolor='k',s=markersize)
        if self.with_outliers=='Categorical':
            threshold=0.68 #
            is_outlier = np.array([self.trace_data['is_outlier__%d'%(i)].mean() for i in range(self.N)])>threshold
            ax.plot(x[is_outlier],y[is_outlier],'ro',ms=15,mfc='None')
        ax.legend(loc=3)
        ax.set_xlabel('log $T$ [d]')
        ax.set_ylabel('log $B$ [$\mu$T]')

        ax.set_title(title)
        
        
       

class model2(Model):

    def run_model(self,Nsample, burnin, outname=None, shared=True):
        if shared:
            xdata=self.x_shared
            ydata=self.bfield_shared
        else:
            xdata=self.log_days
            ydata=self.bfield_log
            
        with pymc3.Model() as self.mymodel:
            #priors
            #alpha = pymc3.Normal('alpha', mu=0, sd=10) unbound
            BoundedNormal = pymc3.Bound(pymc3.Normal, lower=-5, upper=5)
            alpha = BoundedNormal('alpha', mu=0, sd=10)
            
            beta = pymc3.Uniform('beta',lower=-2, upper=3.5)
            
            #alpha_B = pymc3.Normal('alpha_B', mu=0, sd=10)#unbound
            BoundedNormal = pymc3.Bound(pymc3.Normal, lower=-5, upper=5)
            alpha_B = BoundedNormal('alpha_B', mu=0, sd=10)
            beta_B = pymc3.Uniform('beta_B',lower=-2,upper=3.5 )

            #
            prob = self.logistic(alpha * (xdata - beta) + alpha_B * (ydata - beta_B))
        
            step = self.construct_model(prob)
            
            init = pymc3.find_MAP()
            
            if step is None:
                self.trace = pymc3.sample(Nsample,chains=2, n_init=burnin, start=init,  \
                     nuts_kwargs={'target_accept':0.8,'integrator':'two-stage'}, random_seed=SEED)
            else:
                self.trace = pymc3.sample(Nsample,chains=2, n_init=burnin, star=init, step=step, random_seed=SEED)
                
            #save to pandas   
            self.trace_data = pymc3.trace_to_dataframe(self.trace)

    def predict_true(self,  X, Y, params=None):
    
        #compute model
        if params is None:
            params = self.best_params_()

        ndim=len(X.shape)
        
        prob = self.logistic(params['alpha'] * (X - params['beta']) + params['alpha_B'] * (Y - params['beta_B']) )
        
        return prob.eval()
    
    
    def plot_model2D(self,   markersize=100, outname=None, edc='None', add_random=None, ax=None, add_colorbar=True, title=r"Parametric LR model 2"):
    
        """ """
    
        params = self.get_summary().to_dict()
        #print(params)
        #plt.plot()
        xx,yy=np.meshgrid(np.arange(-1.8,3.51,0.05),np.arange(-2,3.51,0.05) )
        Z=self.predict_true(xx,yy)

        if ax is None:
            ax=p.gca()
        im=ax.pcolormesh(xx,yy,Z,cmap=p.cm.Greys,alpha=0.8,edgecolors=edc)
        #p.contourf(xx,yy,Z,cmap=p.cm.Greys,alpha=0.5,edgecolors=edc)
        
        #mylabel = lambda p: r"%.1f$^{+%.1f}_{%.1f}$" %(params[50.0][p],params[84.0][p]-params[50.0][p],params[16.0][p]-params[50.0][p])
        #p.axvline(params[50.0]['beta'],ymin=-2,ymax=params[50.0]['Switch'],ls='-',color='k', label=r"$\beta$="+mylabel('beta'))
        #p.axhline(params[50.0]['Switch'],ls='-.',color='k',label=r"$B_{\rm thr}=$"+mylabel('Switch'))
        
        if add_colorbar:
            #axins1 = inset_axes(ax,
            #            width="50%",  # width = 10% of parent_bbox width
            #            height="5%",  # height : 50%
            #            loc='lower right')
            cax=ax.figure.add_axes([0.1,0.88,0.2,0.03])
            cb=ax.figure.colorbar(im, cax=cax,  orientation='horizontal')
            cb.set_clim([0,1])
            cb.set_ticks([0.2,0.5,0.8])
 
             
  
        if add_random is None:
            x,y = self.log_days,self.bfield_log
        else:
            x,y = self.log_days+add_random[:len(self.outcomes)],self.bfield_log
        
        
       
        ax.scatter(x, y, marker='o',c=self.outcomes*2,label='Rat studies',cmap=p.cm.Paired,edgecolor='k',s=markersize)
        if self.with_outliers=='Categorical':
            threshold=0.68 #
            is_outlier = np.array([self.trace_data['is_outlier__%d'%(i)].mean() for i in range(self.N)])>threshold
            ax.plot(x[is_outlier],y[is_outlier],'ro',ms=15,mfc='None')
       
        ax.legend(loc=3)
        
        ax.set_xlabel('log $T$ [d]')
        ax.set_ylabel('log $B$ [$\mu$T]')
          
        ax.set_title(title)
 
        if outname is not None:
            p.savefig(outname)
 
        
class model3(Model):

    def run_model(self, Nsample, burnin, outname=None, shared=True):
        if shared:
            xdata = self.x_shared
            ydata = self.bfield_shared
        else:
            xdata = self.log_days
            ydata = self.bfield_log
        
        with pymc3.Model() as self.mymodel:
            #priors
            
            BoundedNormal = pymc3.Bound(pymc3.Normal, lower=-9, upper=9)
            alpha = BoundedNormal('alpha', mu=0, sd=10)
            gamma = BoundedNormal('gamma', mu=0, sd=10)
            
            beta  = pymc3.Uniform('beta', lower=-2, upper=3.5)
            #beta_B = beta
            
            switchpoint = pymc3.Uniform('Switch', lower=0, upper=2.5)
            coeff = theano.tensor.switch(ydata<switchpoint, alpha, gamma)
            
            # 1. / [1+exp(-t_duration)]  x 1./[1+exp-t_Bfield]
            prob = self.logistic(coeff * (xdata - beta))  
            
            
            step = self.construct_model(prob)
            
            #mixing steps not allowed
            step = pymc3.step_methods.Metropolis()
            
            init = pymc3.find_MAP()
            
            
            if step is None:
                self.trace = pymc3.sample(Nsample,chains=2, n_init=burnin, start=init,  \
                     nuts_kwargs={'target_accept':0.8,'integrator':'two-stage'}, random_seed=SEED)
            else:
                #step = pymc3.step_methods.Metropolis()#[alpha, gamma, beta, switchpoint, f_out, p_out])
                self.trace = pymc3.sample(Nsample,chains=2,n_init=burnin, start=init,  step=step, random_seed=SEED)
 

            #save to pandas   
            self.trace_data = pymc3.trace_to_dataframe(self.trace)
    
    def predict_true(self, X, Y, params=None ):
        
        #compute model
        if params is None:
            params = self.best_params_()

        p1 = self.logistic(params['alpha'] * (X - params['beta']))
        p2 = self.logistic(params['gamma'] * (X - params['beta']))
          
        prob = p1*(Y<params['Switch']) + p2 * (Y>=params['Switch'])
                  
        return prob.eval()
        
    def plot_model2D(self, Bthreshold, markersize=100, outname=None, edc='None', add_random=None, ax=None, add_colorbar=True, title=r"Parametric LR model3"):
    
        """ """
    
        params = self.get_summary().to_dict()
        #print(params)
        #plt.plot()
        xx,yy=np.meshgrid(np.arange(-1.8,3.51,0.05),np.arange(-2,3.51,0.05) )
        Z=self.predict_true(xx,yy)

        if ax is None:
            ax=p.gca()
        im=ax.pcolormesh(xx,yy,Z,cmap=p.cm.Greys,alpha=0.8,edgecolors=edc)
        #p.contourf(xx,yy,Z,cmap=p.cm.Greys,alpha=0.5,edgecolors=edc)
        
        mylabel = lambda p: r"%.1f$^{+%.1f}_{%.1f}$" %(params[50.0][p],params[84.0][p]-params[50.0][p],params[16.0][p]-params[50.0][p])
        p.axvline(params[50.0]['beta'],ymin=-2,ymax=params[50.0]['Switch'],ls='-',color='k', label=r"$\beta$="+mylabel('beta'))
        p.axhline(params[50.0]['Switch'],ls='-.',color='k',label=r"$B_{\rm thr}=$"+mylabel('Switch'))
        
        if add_colorbar:
            #axins1 = inset_axes(ax,
            #            width="50%",  # width = 10% of parent_bbox width
            #            height="5%",  # height : 50%
            #            loc='lower right')
            cax=ax.figure.add_axes([0.1,0.92,0.2,0.03])
            cb=ax.figure.colorbar(im, cax=cax,  orientation='horizontal')
            cb.set_clim([0,1])
            cb.set_ticks([0.2,0.5,0.8])
 
             
  
        if add_random is None:
            x,y = self.log_days,self.bfield_log
        else:
            x,y = self.log_days+add_random[:len(self.outcomes)],self.bfield_log
        
        
       
        ax.scatter(x, y, marker='o',c=self.outcomes*2,label='Rat studies',cmap=p.cm.Paired,edgecolor='k',s=markersize)
        if self.with_outliers=='Categorical':
            threshold=0.68 #
            is_outlier = np.array([self.trace_data['is_outlier__%d'%(i)].mean() for i in range(self.N)])>threshold
            ax.plot(x[is_outlier],y[is_outlier],'ro',ms=15,mfc='None')
       
        ax.axhline(np.log10(Bthreshold),label="$%d$\mu$T$\simeq B_{\odot}$"  %(Bthreshold),c='k',ls=':',lw=2)
        ax.legend(loc=3)
        
        ax.set_xlabel('log $T$ [d]')
        ax.set_ylabel('log $B$ [$\mu$T]')
          
        ax.set_title(title)
 
        if outname is not None:
            p.savefig(outname)
 
    def plot_posteriors(self,fig, outname=None):
        #ratsAll_comboswitch.pkl

        #with open(chain+'.pkl', 'rb') as buff:
        #    data = pickle.load(buff)  

        #model, trace = data['model'], data['trace']    
      

        fig1=fig

        #p.hist(trace['alpha'],normed=True,bins=50)
        ax1=fig1.add_subplot(411)
        ax1.set_xlabel(r'$\alpha$', labelpad=-5)
        ax1.set_xlim([-5,10])
        ax2=fig1.add_subplot(412)
        ax2.set_xlabel(r'$\beta$', labelpad=-5)
        ax2.set_xlim([0,3]) 
        ax3=fig1.add_subplot(413)
        ax3.set_xlabel(r'$\gamma$', labelpad=-3)
        ax3.set_xlim([-5,5])
        ax4=fig1.add_subplot(414)
        ax4.set_xlim([0,2.5])
        ax4.set_xlabel(r'$\log B_t$', labelpad=1)
        #ax4.axhline(30)

        axs=plot_kde(self.outname,['alpha','beta','gamma', 'Switch'],[ax1,ax2,ax3,ax4],'k',None,lw=2)

        fig1.subplots_adjust(hspace=0.6,bottom=0.1,top=0.95)
        
        if outname is not None:
            p.savefig(outname)
    
class model_linear:
    """
        model = A X + B
    """
          
    def read(self, filename):
        if '.csv' in filename:
            tab = Table.read(filename,format='csv')
        self.x_shared = tab['x'].data - np.mean(tab['x'])
        self.error_y = tab['sigma_y'].data
        self.outcomes = tab['y'].data - np.mean(tab['y'])
        self.data = tab
        self.N = np.shape(tab['x'])[0]
        
    def run_model(self, Nsample, burnin, outname=None, shared=False, with_outliers=False):
        
        xdata = self.x_shared 
         
        
        with pymc3.Model() as self.mymodel:
            #priors
            
            BoundedNormal = pymc3.Bound(pymc3.Normal, lower=-9, upper=9)
            alpha = BoundedNormal('alpha', mu=0, sd=10)
            beta  = pymc3.Uniform('beta', lower=-50, upper=50)
           
            
            yobs = theano.shared(np.asarray(self.outcomes, dtype=theano.config.floatX))
            sigma_y = theano.shared(np.asarray(self.error_y, dtype=theano.config.floatX))
                
            yest_in = alpha * xdata+ beta
            
            sigma_int = sigma_y * 0
             
            step = None
            if with_outliers:
                 ## Define weakly informative priors for the mean and variance of outliers
                yest_out = pymc3.Normal('yest_out', mu=0, sd=10 )
                sigma_y_out = pymc3.HalfNormal('sigma_y_out', sd=100)

                ## Define Bernoulli inlier / outlier flags according to a hyperprior
                ## fraction of outliers, itself constrained to [0, .5] for symmetry
                frac_outliers = pymc3.Uniform('frac_outliers', lower=0.0, upper=0.75)
                is_outlier = pymc3.Bernoulli('is_outlier', p=frac_outliers, shape=self.N)

                # Set up normal distributions that give us the logp for both distributions
                inliers = pymc3.Normal.dist(mu=yest_in, sd=sigma_y).logp(yobs)
                outliers = pymc3.Normal.dist(mu=yest_out, sd=sigma_y + sigma_y_out).logp(yobs)
               
                # Build custom likelihood, a potential will just be added to the logp and can thus function
                # like a likelihood that we would add with the observed kwarg.
                o = pymc3.Potential('outcome', ((1 - is_outlier) * inliers).sum() + (is_outlier * outliers).sum())
            
                #mixing steps not allowed
                step = pymc3.step_methods.Metropolis([alpha, beta, sigma_int, sigma_y_out, yest_out,  frac_outliers, is_outlier])
            else:
                #mixing steps not allowed
                #step = pymc3.step_methods.Metropolis([alpha, beta, sigma_int])
                o = pymc3.Normal('outcome', mu=yest_in, sd=np.sqrt(sigma_y**2+sigma_int**2), observed=self.outcomes) #likelihood
             
            
            
            init = pymc3.find_MAP()
            if step is not None:
                self.trace = pymc3.sample(Nsample,chains=2,n_init=burnin, start=init,  step=step, random_seed=SEED)
            else:
                self.trace = pymc3.sample(Nsample,chains=2,n_init=burnin, start=init, random_seed=SEED)
                
            #save to pandas   
            self.trace_data = pymc3.trace_to_dataframe(self.trace)

    def plot_posteriors(self,outname=None):
        """
        """
        raise NotImplementedError
        
    def plot_trace(self):
        pymc3.traceplot(self.trace,combined=False)
        
   
    def plot_corner(self, Nsig=3):
        try:
           import corner
        except:
            raise "corner package required"
                     
        #corner.corner(trace)
        r=np.ones(len(self.trace_data.keys()))*0.97
        perc = np.percentile(self.trace_data,(16,84),axis=(0,))#1sigma
        m = np.median(self.trace_data,axis=0)
        s = perc[1,:] - perc[0,:]
        corner.corner(self.trace_data,range=zip(m-Nsig*s,m+Nsig*s),truths=m )


