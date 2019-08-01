# coding=utf-8
import os, argparse
from os.path import join,dirname
import numpy as np
np.random.seed(1234)
rand_offs= np.random.normal(0,0.03,size=40)

from matplotlib import pyplot as plt

try:
    from .logistic import Modelling, scikit_regression
except Exception:
    try:
       from logistic import Modelling, scikit_regression
    except Exception:
       import Modelling, scikit_regression 
    
dirpath = os.path.dirname(__file__)
dpath = join(dirname(dirpath),"data/") 

file_humans = dpath + 'Halgamuge2013_Table4.tex'
file_rats   = dpath + 'Jahandideh_Table.tex'  
 
file_macros = 'bfield_paper_macros.tex'

Bthreshold = 45 #

def paper_models(Nsample=-1, with_outliers='default', path=''):

    if  with_outliers is None:
        
        mH1=Modelling.model1(file_humans,outname=path+'humansAll_model1', Nsample=Nsample, with_outliers=with_outliers)  

        r1=Modelling.model1(file_rats, outname=path+'ratsAll_model1', Nsample=Nsample, with_outliers=with_outliers)
        r1b=Modelling.model1(file_rats,outname=path+'ratsBlt50_model1', threshold=[0,Bthreshold], Nsample=Nsample, with_outliers=with_outliers)
        r1c=Modelling.model1(file_rats,outname=path+'ratsBgt50_model1', threshold=[Bthreshold,5e3], Nsample=Nsample, with_outliers=with_outliers)
        r3=Modelling.model3(file_rats, outname=path+'ratsAll_model3', Nsample=Nsample)

    else:        
        mH1=Modelling.model1(file_humans,outname=path+'humansAll_model1_'+with_outliers, Nsample=Nsample, with_outliers=with_outliers)  

        r1=Modelling.model1(file_rats,outname= path+'ratsAll_model1_'+with_outliers, Nsample=Nsample, with_outliers=with_outliers)
        r1b=Modelling.model1(file_rats,outname=path+'ratsBlt50_model1_'+with_outliers, threshold=[0,Bthreshold], Nsample=Nsample, with_outliers=with_outliers)
        r1c=Modelling.model1(file_rats,outname=path+'ratsBgt50_model1_'+with_outliers, threshold=[Bthreshold,5e3], Nsample=Nsample, with_outliers=with_outliers)
        r3=Modelling.model3(file_rats,outname= path+'ratsAll_model3_'+with_outliers, Nsample=Nsample, with_outliers=with_outliers)

    return mH1,  r1, r1b, r1c, r3


def run_Humans(filename, Nsample, with_outliers=None, path=''):

    if with_outliers is None:
        mH1=Modelling.model1(filename,outname=path+'humansAll_model1', Nsample=Nsample)  
        mH3=Modelling.model3(filename,outname=path+'humansAll_model3', Nsample=Nsample)
    #
    else:
        mH1=Modelling.model1(filename,outname=path+'humansAll_model1_'+with_outliers, Nsample=Nsample, with_outliers=with_outliers)  
        mH3=Modelling.model3(filename,outname=path+'humansAll_model3_'+with_outliers, Nsample=Nsample, with_outliers=with_outliers)

    return mH1,  mH3
    
def run_Rats(filename, Nsample, with_outliers=None, path=''):
    
    if with_outliers is None:
        r1=Modelling.model1(filename,outname=path+'ratsAll_model1', Nsample=Nsample)
        r1b=Modelling.model1(filename,outname=path+'ratsBlt50_model1', threshold=[0,Bthreshold], Nsample=Nsample)
        r1c=Modelling.model1(filename,outname=path+'ratsBgt50_model1', threshold=[Bthreshold,5e3], Nsample=Nsample)
        r3=Modelling.model3(filename,outname=path+'ratsAll_model3', Nsample=Nsample)
    #
    else:
        r1=Modelling.model1(filename,outname=path+'ratsAll_model1_'+with_outliers, Nsample=Nsample, with_outliers=with_outliers)
        r1b=Modelling.model1(filename,outname=path+'ratsBlt50_model1_'+with_outliers, threshold=[0,Bthreshold], Nsample=Nsample, with_outliers=with_outliers)
        r1c=Modelling.model1(filename,outname=path+'ratsBgt50_model1_'+with_outliers, threshold=[Bthreshold,5e3], Nsample=Nsample, with_outliers=with_outliers)
        r3=Modelling.model3(filename,outname=path+'ratsAll_model3_'+with_outliers, Nsample=Nsample, with_outliers=with_outliers)

    return r1, r1b, r1c, r3
    
 
def make_histogram_figure(table1,table2, outpdf='Bfield_histogram.pdf'):

    hist=Modelling.Histogram_Bfield(humans=table1,rats=table2)
    hist.make_plot(outpdf=outpdf,verbose=True)
    print("Histogram: P-val",hist.Pval)

    return hist
    
def make_macros(humans, rats, Bhist, fname=None):

    mH1,  [r1, r1b, r1c, r3] = humans, rats
    r3.print_intervals(ci=0.68)
    
    my_str = r"""
\newcommand{\Nhums}{%d}
\newcommand{\Nrats}{%d}
\newcommand{\Ntot}{%d}
\newcommand{\HistPval}{%.2f}

%%95CL
%%Human Model A
\newcommand{\humansModelOneType}{%s}
\newcommand{\humansModelOneAlpha}{%.1f} 
\newcommand{\humansModelOneAlphaErr}{$^{+%.1f}_{%.1f}$}
\newcommand{\humansModelOneBeta}{%.1f}
\newcommand{\humansModelOneBetaErr}{$^{+%.1f}_{%.1f}$}

\newcommand{\humansModelOneBetaDays}{%d}

\newcommand{\humansModelOneWAIC}{%.1f}
\newcommand{\humansModelOneLOO}{%.1f}
\newcommand{\humansModelOneDIC}{%.1f}

\newcommand{\humansModelOneALPHA}{%s}
\newcommand{\humansModelOneBETA}{%s}
 
""" % (mH1.N,
    r1.N,
    mH1.N+r1.N,
    Bhist.Pval,
    mH1.with_outliers,
    mH1.summary[50.0]['alpha'], mH1.summary[97.5]['alpha']-mH1.summary[50.0]['alpha'],  mH1.summary[2.5]['alpha']-mH1.summary[50.0]['alpha'],
    mH1.summary[50.0]['beta'],  mH1.summary[97.5]['beta']-mH1.summary[50.0]['beta'],    mH1.summary[2.5]['beta']-mH1.summary[50.0]['beta'],
    10**mH1.summary[50.0]['beta'],
    mH1.WAIC,mH1.LOO,mH1.DIC, 
    mH1.my_range['alpha'],mH1.my_range['beta'],
 )
    
    if mH1.with_outliers=='Robust_LR05' or mH1.with_outliers=='Categorical':
        my_str += r"""
\newcommand{\humansModelOnePI}{%s}
"""   % (mH1.my_range['f_out'])
        
    elif mH1.with_outliers =='Robust_LR':
        my_str += r"""
\newcommand{\humansModelOnePI}{%s}
\newcommand{\humansModelOnePOUT}{%s}
"""   % (mH1.my_range['f_out'], mH1.my_range['p_out'])
    else:
        pass
        
    my_str += r"""
%%Rat Model A
\newcommand{\ratsModelOneType}{%s}
\newcommand{\ratsModelOneAlpha}{%.1f}
\newcommand{\ratsModelOneAlphaErr}{$^{+%.1f}_{%.1f}$}

\newcommand{\ratsModelOneWAIC}{%.1f}
\newcommand{\ratsModelOneLOO}{%.1f}
\newcommand{\ratsModelOneDIC}{%.1f}

\newcommand{\ratsModelOneALPHA}{%s}
\newcommand{\ratsModelOneBETA}{%s}



%%Rat Model B
%%68CL
\newcommand{\ratsModelThreeType}{%s}
\newcommand{\ratsModelThreeSwitch}{%.1f}
\newcommand{\ratsModelThreeSwitchErr}{$^{+%.1f}_{%.1f}$}

\newcommand{\ratsModelThreeWAIC}{%.1f}
\newcommand{\ratsModelThreeLOO}{%.1f}
\newcommand{\ratsModelThreeDIC}{%.1f}


\newcommand{\ratsModelThreeALPHA}{%s}
\newcommand{\ratsModelThreeBETA}{%s}
\newcommand{\ratsModelThreeGAMMA}{%s}
\newcommand{\ratsModelThreeSWITCH}{%s}

""" % (
    r1.with_outliers,
    r1.summary[50.0]['alpha'],  r1.summary[97.5]['alpha']-r1.summary[50.0]['alpha'],    r1.summary[2.5]['alpha']-r1.summary[50.0]['alpha'],
    r1.WAIC,r1.LOO,r1.DIC, 
    r1.my_range['alpha'],r1.my_range['beta'],
  
    r3.with_outliers,
    r3.summary[50.0]['Switch'], r3.summary[84.0]['Switch']-r3.summary[50.0]['Switch'],  r3.summary[16.0]['Switch']-r3.summary[50.0]['Switch'],
    r3.WAIC,r3.LOO,r3.DIC, 
    r3.my_range['alpha'],r3.my_range['beta'],r3.my_range['gamma'],r3.my_range['Switch']
   )

    if r3.with_outliers=='Robust_LR05' or r3.with_outliers=='Categorical':
        my_str += r"""
\newcommand{\ratsModelThreePI}{%s}
"""   % (r3.my_range['f_out'])
        
    elif r3.with_outliers =='Robust_LR':
        my_str += r"""
\newcommand{\ratsModelThreePI}{%s}
\newcommand{\ratsModelThreePOUT}{%s}
"""   % (r3.my_range['f_out'], r3.my_range['p_out'])
    else:
        pass

    if r1.with_outliers=='Robust_LR05' or r1.with_outliers=='Categorical':
        my_str += r"""
\newcommand{\ratsModelOnePI}{%s}
"""   % (r1.my_range['f_out'])
        
    elif r1.with_outliers =='Robust_LR':
        my_str += r"""
\newcommand{\ratsModelOnePI}{%s}
\newcommand{\ratsModelOnePOUT}{%s}
"""   % (r1.my_range['f_out'], r1.my_range['p_out'])
    else:
        pass
    
    my_str +="""
%%Rat model A below threshold
\\newcommand{\\ratsModelOneBthreshold}{%d}
\\newcommand{\\ratsModelOneBAlpha}{%.1f}
\\newcommand{\\ratsModelOneBAlphaErr}{$^{+%.1f}_{%.1f}$}

\\newcommand{\\ratsModelOneBBeta}{%.1f}
\\newcommand{\\ratsModelOneBBetaErr}{$^{+%.1f}_{%.1f}$}

\\newcommand{\\ratsModelOneBALPHA}{%s}
\\newcommand{\\ratsModelOneBBETA}{%s}

%%Rat model A above threshold
\\newcommand{\\ratsModelOneCAlpha}{%.1f}
\\newcommand{\\ratsModelOneCAlphaErr}{$^{+%.1f}_{%.1f}$}

\\newcommand{\\ratsModelOneCBeta}{%.1f}
\\newcommand{\\ratsModelOneCBetaErr}{$^{+%.1f}_{%.1f}$}

\\newcommand{\\ratsModelOneCALPHA}{%s}
\\newcommand{\\ratsModelOneCBETA}{%s}
""" % (  
    Bthreshold,
    r1b.summary[50.0]['alpha'],  r1b.summary[97.5]['alpha']-r1b.summary[50.0]['alpha'],    r1b.summary[2.5]['alpha']-r1b.summary[50.0]['alpha'],
    r1b.summary[50.0]['beta'],  r1b.summary[97.5]['beta']-r1b.summary[50.0]['beta'],    r1b.summary[2.5]['beta']-r1b.summary[50.0]['beta'],
    
    r1b.my_range['alpha'],r1b.my_range['beta'],
   
    r1c.summary[50.0]['alpha'],  r1c.summary[97.5]['alpha']-r1c.summary[50.0]['alpha'],    r1c.summary[2.5]['alpha']-r1c.summary[50.0]['alpha'],
    r1c.summary[50.0]['beta'],  r1c.summary[97.5]['beta']-r1c.summary[50.0]['beta'],    r1c.summary[2.5]['beta']-r1c.summary[50.0]['beta'],
    
    r1c.my_range['alpha'],r1c.my_range['beta']
    )
    if r1b.with_outliers=='Robust_LR05' or r1b.with_outliers=='Categorical':
        my_str += r"""
\newcommand{\ratsModelOneBPI}{%s}
"""   % (r1b.my_range['f_out'])
        
    elif r1b.with_outliers =='Robust_LR':
        my_str += r"""
\newcommand{\ratsModelOneBPI}{%s}
\newcommand{\ratsModelOneBPOUT}{%s}
"""   % (r1b.my_range['f_out'], r1b.my_range['p_out']) 
    else:
        pass
        
    if r1c.with_outliers=='Robust_LR05' or r1c.with_outliers=='Categorical':
        my_str += r"""
\newcommand{\ratsModelOneCPI}{%s}
"""   % (r1c.my_range['f_out'])
        
    elif r1c.with_outliers =='Robust_LR':
        my_str += r"""
\newcommand{\ratsModelOneCPI}{%s}
\newcommand{\ratsModelOneCPOUT}{%s}
"""   % (r1c.my_range['f_out'], r1c.my_range['p_out']) 
    else:
        pass
 
    if fname is not None:
        f = open(fname,'w')
        f.write(my_str)
        f.close()
    
    return my_str
    
def run_all(read=False, with_outliers=None, path=''):

    if read==False:
        mH1,  mH3 = run_Humans(file_humans, Nsample=15e3, with_outliers=with_outliers, path=path)
        r1, r1b, r1c, r3 = run_Rats(file_rats,   Nsample=25e3,with_outliers=with_outliers, path=path)
    else:
        mH1,   mH3 = run_Humans(file_humans, Nsample=-1, with_outliers=with_outliers, path=path)
        r1, r1b, r1c,  r3 = run_Rats(file_rats,   Nsample=-1, with_outliers=with_outliers, path=path)
    
    return [mH1,  mH3], [r1, r1b, r1c, r3]


def main(run, outpath=None,summary=False,figure=None,  with_outliers="Robust_LR"):
    '''
    run: boolean 
        to run MCMC code (pymc3)
        
    output: str
        output path root directory
        
    summary: boolean [default:False]
        print summary on screen
        
    figure:
        None [default]  to make macros and all figure
        0 to do no macros and no figures
        -1 to make macros
        1,2,3,4,5 to make specific figure
    
    with_outliers: 
        'Robust_LR' [default] robust LR with outlier rejection
        'Robust_LR05' robust LR with pout=0.5
        'normal' no outlier rejection
        'Categorical' experimenal // not supported
        
    '''
    
    if outpath is None:
        raise Exception("Please specify outpath to save outputs")
    else:
        print("Will save data on outpath="+outpath)
        if os.path.isdir(outpath) is False:
            os.system('mkdir '+outpath)
        
    
    if with_outliers == 'Robust_LR':
        path = join(outpath, 'mRobust_LR/')
    elif with_outliers == 'Robust_LR05':
        path = join(outpath, 'mRobust_LR05/')
    elif with_outliers == 'Categorical':
        path = join(outpath, 'mCategorical/')
    else:
        path = join(outpath, 'mDefault/')
        
    if os.path.isdir(path) is False:
        os.system('mkdir '+path)
        
    #with_outliers:
    #   Mixture | Mixture_05 | Robust_LR | Robust_LR05
    if figure is None or figure == -1 or figure==1:
        #figure 1:
        f=plt.figure(1,figsize=(10,8))
        f.clf()
        Bhist = make_histogram_figure(file_humans,file_rats, outpdf = join(outpath, 'Figure1.pdf'))

    if run:
        #run_all
        models_humans, models_rats = run_all(with_outliers=with_outliers,path=path)
        [mH1,  mH3], [r1, r1b, r1c,  r3] = models_humans, models_rats
    else:    
        #read output
        mH1, r1, r1b, r1c, r3 = paper_models(with_outliers=with_outliers,path=path)
        
    
    r3.print_intervals(ci=0.68)

    sk = scikit_regression.scikit(mH1,r3,Bthreshold)
    
    if figure is None or figure==2:    
        #figure 2 variant compact
        f=plt.figure(2,figsize=(14,8))
        f.clf()
        ax1=f.add_axes([0.06,0.1,0.43,0.2])
        mH1.plot_wtime(title='',outname=None)
        
        ax2=f.add_axes([0.06,0.3,0.43,0.67])
        mH1.plot_model2D(add_random=rand_offs, title="")
        ax2.get_xaxis().set_ticklabels([])
        ax2.set_xlabel('')
        
        ax3=f.add_axes([0.55,0.3,0.43,0.67])
        C=sk.SVC_regression(dataset='humans', add_random=rand_offs)    
        #ax3.get_yaxis().set_ticklabels([])
        #ax3.set_ylabel('')
        
        f.savefig(join(outpath,'Figure2.pdf'))
        
    if figure==22:
        #figure 2 variant compact
        f=plt.figure(2,figsize=(14,8))
        f.clf()
        ax1=f.add_axes([0.06,0.1,0.43,0.2])
        mH1.plot_wtime(title='',outname=None)
        
        ax2=f.add_axes([0.06,0.3,0.43,0.67])
        mH1.plot_model2D(add_random=rand_offs, title="")
        ax2.get_xaxis().set_ticklabels([])
        ax2.set_xlabel('')
        
        ax3=f.add_axes([0.55,0.3,0.43,0.67])
        C=sk.SVR_regression(dataset='humans', add_random=rand_offs)    
        #ax3.get_yaxis().set_ticklabels([])
        #ax3.set_ylabel('')
        
        f.savefig(join(outpath,'Figure2b.pdf'))
   
    if figure is None or figure==3:    
        #figure 3
        #r1.plot_wtime(title='Rats studies', outname= '../figs/ratsAll_model1.pdf')
        
        f=plt.figure(3,figsize=(8,8))
        f.clf()
        ax1=plt.subplot(211)
        r1c.plot_wtime(title='Rats B$>=$%d $\mu$T' % (Bthreshold), outname = None, ycust=[-0.9,1.5], text=False)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set_xlabel('')
        
        ax2=plt.subplot(212,sharex=ax1)
        r1b.plot_wtime(title='Rats B$<$%d $\mu$T' %(Bthreshold), outname = None,  ycust=[-0.9,1.5], text=False)
        
       
        f.savefig(join(outpath,'Figure3.pdf'))

        
    if figure is None or figure==4:
        f=plt.figure(4,figsize=(14,8))
        f.clf()
        ax1=plt.subplot(121)
        r3.plot_model2D(Bthreshold,add_random=rand_offs, title="")
                
        ax3=f.add_subplot(122)
        C=sk.SVC_regression(dataset='rats',add_random=rand_offs)
        f.subplots_adjust(wspace=0.15,left=0.05,right=0.95)
        f.tight_layout()
        f.savefig(join(outpath,'Figure4.pdf'))
   
    if figure is None or figure==5:    
        #figure 5
        f=plt.figure(5,figsize=(5,8))
        f.clf()
        #mH3.plot_posteriors(outname='../figs/humansAll_comboswitch_posteriors.pdf')
        r3.plot_posteriors(f,outname=join(outpath,'Figure5.pdf'))
        f.tight_layout()   
        
    #######MACROS
    if figure == -1 or figure is None:
        make_macros(mH1,[r1,r1b,r1c,r3], Bhist, fname=join(outpath,file_macros))
     
     
    if summary:
        #table1 
        mH1.summary
        
        #table2
        r1.summary
        r1b.summary
        r1c.summary
        r3.summary
        
        #table3
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("outpath",help=" to specify where to put the outputs", default='.')
    parser.add_argument("--format", help=" default | Robust_LR | Robust_LR05 | Categorical ", default='Robust_LR')
    parser.add_argument("--figure", help=" to specify which figure from paper to remake ", type=int)
    parser.add_argument("--read", help="to read previous runs", action='store_true')
    parser.add_argument("--summary", help="to print summary of results", action='store_true')
    
    args = parser.parse_args()
    print(args)
    #print(args.read)
    make_run=args.read==False
    main(run=make_run, outpath=args.outpath, figure=args.figure, with_outliers=args.format)
    
