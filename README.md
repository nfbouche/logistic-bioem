# logistic-bioem
Logistic Regression on Melatonin Biodata by  Nicolas Bouché (nicolas.bouche@univ-lyon1.fr).

This is the algorithm used to perform the analysis in Bouché & McConway, Bioelectromagnetics 2019

It uses PyMC3 and requires python2.7 or 3.5 (in a virtual environment). See INSTALL for instruction.

The figures of the paper can be made using this code as follows:

# Run from ipython from 

ipython
> from logistic import run_all
> run_all(run=True,output='path_to_save/')

# Run as script:
```
python logistic/run_all.py path_to_save 

python logistic/run_all.py path_to_save --format Robust_LR      #to use robust LR
python logistic/run_all.py path_to_save --format Robust_LR05    #to use robust LR with p_out=0.5
python logistic/run_all.py path_to_save --format default        #to turnoff robust_LR
```
# Bugs & issues

Please send bug report and/or issues with installation to nicolas.bouche@univ-lyon1.fr
