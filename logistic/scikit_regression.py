from sklearn import datasets, neighbors, linear_model, svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from matplotlib import pyplot as p
import numpy as np

from . import Modelling


#Bthreshold=45
class scikit:

    def __init__(self, h,r,Bthreshold):
        self.humans=h
        self.rats=r
        self.Bthreshold=Bthreshold
        
        #file_humans='../Halgamuge2013_Table4.tex'
        #humans=Modelling.model2(file_humans,outname='ratH_model2', Nsample=15e3,with_outliers=False)
        #humans=Modelling.model2(file_humans,outname='humansAll_model2', Nsample=-1)
        self.Hdata=np.array(list(zip(self.humans.log_days,self.humans.bfield_log)))

        #file_rats = '../Jahandideh_Table.tex'
        ##rats=Modelling.model3(file_rats,outname='ratsAll_model4', Nsample=15e3,with_outliers=False)
        #rats=Modelling.model3(file_rats,outname='ratsAll_model3', Nsample=-1)
        self.Rdata=np.array(list(zip(self.rats.log_days,self.rats.bfield_log)))
        

        self.allOutcomes=np.concatenate((self.rats.outcomes*2,self.humans.outcomes*2),axis=0)
        self.allData = np.concatenate((self.Rdata,self.Hdata),axis=0)
        
    def plot2D_humans(self,markersize=100,add_random=None,ax=None):

        if ax is None:
            ax=p.gca()

        if add_random is None:
          ax.scatter(self.Hdata[:,0],self.Hdata[:,1],marker='s',c=self.humans.outcomes*2,label='Human studies',cmap=p.cm.Paired,edgecolor='k',s=markersize)
        else:
          ax.scatter(self.Hdata[:,0]+add_random[:len(self.humans.outcomes)],self.Hdata[:,1],marker='s',c=self.humans.outcomes*2,label='Human studies',cmap=p.cm.Paired,edgecolor='k',s=markersize)

        ax.legend(loc=3)
        ax.set_xlabel('log $T$ [d]')
        ax.set_ylabel('log $B$ [$\mu$T]')

    def plot2D_rats(self,markersize=100, band=False,add_random=None,ax=None):
        
        if ax is None:
            ax=p.gca()
                
        if add_random is None:
          ax.scatter(self.Rdata[:,0],self.Rdata[:,1],marker='o',c=self.rats.outcomes*2,label='Rat studies',cmap=p.cm.Paired,edgecolor='k',s=markersize)
        else:
          ax.scatter(self.Rdata[:,0]+add_random[:len(self.rats.outcomes)],self.Rdata[:,1],marker='o',c=self.rats.outcomes*2,label='Rat studies',cmap=p.cm.Paired,edgecolor='k',s=markersize)

        ax.axhline(np.log10(self.Bthreshold),label='$%d$\mu$T$\simeq B_{\odot}$' %(self.Bthreshold),c='k',ls=':',lw=2)
        #rats.summary[84.0]['Switch']-rats.summary[50.0]['Switch'],  rats.summary[16.0]['Switch']-rats.summary[50.0]['Switch']
        if band:
            ax=p.gca()
            ax.add_patch(p.Rectangle((-1.5,self.rats.summary[16.0]['Switch']),5,self.rats.summary[84.0]['Switch']-self.rats.summary[16.0]['Switch'],fill=True,fc='grey',alpha=0.35))

        ax.legend(loc=3)
        ax.set_xlabel('log $T$ [d]')
        ax.set_ylabel('log $B$ [$\mu$T]')
      
    def plot2D_combo(self,markersize=100):
          p.scatter(self.allData[:,0],self.allData[:,1],marker='s',c=self.allOutcomes,label='Humans \&\ rats',cmap=p.cm.Paired,edgecolor='k',s=markersize)
          #rats.summary[84.0]['Switch']-rats.summary[50.0]['Switch'],  rats.summary[16.0]['Switch']-rats.summary[50.0]['Switch']
          ax=p.gca()
          ax.add_patch(Rectangle((-1.5,self.rats.summary[16.0]['Switch']),5,self.rats.summary[84.0]['Switch']-self.rats.summary[16.0]['Switch'],fill=True,fc='grey',alpha=0.35))
          p.legend()
          p.xlabel('log $T$ [d]')
          p.ylabel('log $B$ [$\mu$T]')

        
    def LR_regression(self,dataset,solver='liblinear',C=5,figure=None):    
        """
             dataset  "humans" | "rats" | "combined"
        """
        reg=LogisticRegression(C=C,solver=solver)
        
        if figure is not None:
            p.figure(figure)
            p.clf()
        
        if dataset=='humans':
            reg.fit(self.Hdata,self.humans.outcomes*2)
        elif dataset=='rats':
            reg.fit(self.Rdata,self.rats.outcomes*2)
        elif dataset=='combined':
            reg.fit(self.allData,self.allOutcomes)
            
        xx,yy=np.meshgrid(np.arange(-1.5,3.5,0.1),np.arange(-2,3.5,0.1))
        Z=reg.predict(np.c_[xx.ravel(),yy.ravel()])
        Z=Z.reshape(xx.shape)

        p.pcolormesh(xx,yy,Z,cmap=p.cm.Paired,alpha=0.5)
        p.title('Non-parametric LR')

        if dataset=='humans':
            plot2D_humans()
        elif dataset=='rats':
            plot2D_rats(band=False)
        elif dataset=='combined':
            plot2D_humans()
            plot2D_rats(band=True)
            #plot2D_combo()

    def SVR_regression(self,dataset,kernel='linear',C=None,figure=None, scoring='accuracy',add_random=None, ax=None, add_colorbar=True):
        """
             dataset  "humans" | "rats" | "combined"
             kernel   'rbf' [default] | 'sigmoid' 
             method "SVC" | "SVR" |
             decision_function_shape 'ovo' | 'ovr' [default]
             scoring f1 | average_precision | accuracy | balanced_accuracy | 
                    https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        """
       
        if figure is not None:
            p.figure(figure)
            p.clf()
      
        #gridsearch
        parameters =  {'kernel':[kernel], 'gamma': ['auto'], 'C': np.logspace(0,1,25)}
        cls = svm.SVR
        
        if dataset=="humans":
            xdata,ydata = self.Hdata,self.humans.outcomes*2
        elif dataset=='rats':
            xdata,ydata = self.Rdata,self.rats.outcomes*2
        elif dataset=='combined':
            xdata,ydata = self.allData,self.allOutcomes
     
        if C is None:
            #optimize parameters
            clf = GridSearchCV(cls(), parameters, cv=5 )
            clf.fit(self.Hdata,self.humans.outcomes*2)
            print("Best SVC paramters",clf.best_params_,clf.best_score_)

            #using SVC
            svc = cls(kernel=clf.best_params_['kernel'],C=clf.best_params_['C'],gamma=clf.best_params_['gamma']) #,de 
            C=clf.best_params_['C']
            
        else:
            svc = cls(kernel=kernel, C=C, gamma='auto')
         
        #svc = cls(kernel=kernel, C=C)
        
        svc.fit(xdata, ydata)
        
        xx,yy=np.meshgrid(np.arange(-1.8,3.51,0.1),np.arange(-2,3.51,0.1))
        Z=svc.predict(np.c_[xx.ravel(),yy.ravel()])
        Z=Z.reshape(xx.shape)
        
        if ax is None:
            ax=p.gca()
        
        #renormalize to 0,1
        Z=(Z-np.min(Z))/(np.max(Z)-np.min(Z))
        
        im=ax.pcolormesh(xx,yy,Z,cmap=p.cm.Greys)# data was multiplied by 2
        ct=ax.contour(xx,yy,Z,[0.5],c='k',ls='-.', label='P=0.5')
        
        if add_colorbar:
                #axins1 = inset_axes(ax,
                #            width="50%",  # width = 10% of parent_bbox width
                #            height="5%",  # height : 50%
                #            loc='lower right')
                cax=ax.figure.add_axes([0.1,0.88,0.18,0.03])
                cb=ax.figure.colorbar(im, cax=cax,  orientation='horizontal')
                cb.set_clim([0,1])
                cb.set_ticks([0.2,0.5,0.8])
                #cb.add_lines(0.5)
     
     
        
        #ax.set_title("""Non-parametric SVR(kernel='%s',C={%.1f})""" %(kernel, C) )

        if dataset=='humans':
            self.plot2D_humans(add_random=add_random,ax=ax)
        elif dataset=='rats':
            self.plot2D_rats(add_random=add_random,ax=ax)
        elif dataset=='combined':
            self.plot2D_humans(markersize=130)
            self.plot2D_rats(markersize=90)
            #plot2D_combo()

            
        return C
        
    def SVC_regression(self,dataset, C=None,figure=None,  scoring='accuracy', add_random=None):
        """
             dataset  "humans" | "rats" | "combined"
             kernel   'rbf' [default] | 'sigmoid' 
             method "SVC" | "SVR" |
             decision_function_shape 'ovo' | 'ovr' [default]
             scoring f1 | average_precision | accuracy | balanced_accuracy | 
                    https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        """
       
        if figure is not None:
            p.figure(figure)
            p.clf()
        
       
        #gridsearch
        parameters =  {'kernel':['rbf'], 'gamma': ['auto'], 'C': np.logspace(0,1,25)}
        cls = svm.SVC
        
        
        if dataset=="humans":
            xdata,ydata = self.Hdata,self.humans.outcomes*2
            ymin=-1
            ymax=2.5
        elif dataset=='rats':
            xdata,ydata = self.Rdata,self.rats.outcomes*2
            ymin=-0.2
            ymax=3.2
        elif dataset=='combined':
            xdata,ydata = self.allData,self.allOutcomes
            ymin=-2
            ymax=3.51
        
        if C is None:
            #optimize parameters
            clf = GridSearchCV(cls(), parameters, cv=5, scoring=scoring)
            clf.fit(self.Hdata,self.humans.outcomes*2)
            print("Best SVC paramters",clf.best_params_,clf.best_score_)
            
            #using SVC
            svc= cls(kernel=clf.best_params_['kernel'],C=clf.best_params_['C'],gamma=clf.best_params_['gamma']) #,decis 
            
            C=clf.best_params_['C']
            
        else:
            svc = cls(kernel='rbf', C=C, gamma='auto')
        
        
        
        #print(svc.__doc__)
        #svc= svm.SVR(kernel=clf.best_params_['kernel'],C=clf.best_params_['C'])#for linear regression 
        svc.fit(xdata, ydata)
        
        xx,yy=np.meshgrid(np.arange(-1.8,3.51,0.1),np.arange(ymin,ymax,0.1))
        Z=svc.predict(np.c_[xx.ravel(),yy.ravel()])
        Z=Z.reshape(xx.shape)
        p.pcolormesh(xx,yy,Z,cmap=p.cm.Paired,alpha=0.5)
        p.ylim([-2,3.51])

        #p.title("""Non-parametric SVC(kernel='rbf',C={%.1f})""" %(C) )

        if dataset=='humans':
            self.plot2D_humans(add_random=add_random)
        elif dataset=='rats':
            self.plot2D_rats(add_random=add_random)
        elif dataset=='combined':
            self.plot2D_humans(markersize=130)
            self.plot2D_rats(markersize=90)
            #plot2D_combo()
            
        return C

