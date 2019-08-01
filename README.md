# logistic-bioem
Logistic Regression on Melatonin Biodata by  Nicolas Bouché (nicolas.bouche@univ-lyon1.fr).

This is the algorithm used to perform the analysis in Bouché & McConway, Bioelectromagnetics 2019

It uses PyMC3 and requires python2.7 or 3.5 (in a virtual environment). See INSTALL for instruction.

The figures of the paper can be made using the script below.

See the INSTALL instructions

# Run with 

"run_all(run=True,output='path_to_save/')"

cd logistic-bioem;

ipython
```
from logistic import run_all
run_all(run=True,output='paper/')
```

# Run as script:

to recreate the figures from the paper use
```
cd logistic-bioem;
python logistic/run_all.py paper/
```

In addition, it is possible to experiment with the outlier rejection as follows
```
python logistic/run_all.py path_to_save --format Robust_LR      #to use robust LR
python logistic/run_all.py path_to_save --format Robust_LR05    #to use robust LR with p_out=0.5
python logistic/run_all.py path_to_save --format default        #to turnoff robust_LR
```
# Bugs & issues

Please send bug report and/or issues with installation to nicolas.bouche@univ-lyon1.fr
