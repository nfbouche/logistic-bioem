# logistic-bioem
Logistic Regression on Biodata

Algorithm used to perform the analysis in Bouché, McConway 2019

Figures can be made using this script as follows:

# Run from ipython from 

ipython
> from logistic import run_all
> run_all(run=True,output='path_to_save/')

# Run as script:

python logistic/run_all.py path_to_save 

python logistic/run_all.py path_to_save --format Robust_LR      #to use robust LR
python logistic/run_all.py path_to_save --format Robust_LR05    #to use robust LR with p_out=0.5
python logistic/run_all.py path_to_save --format default        #to turnoff robust_LR
