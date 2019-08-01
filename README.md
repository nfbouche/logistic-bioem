# logistic-bioem
Logistic Regression on Melatonin Biodata by  Nicolas Bouché (nicolas.bouche@univ-lyon1.fr).

This is the algorithm used to perform the analysis in Bouché & McConway, Bioelectromagnetics 2019

It uses PyMC3 and requires python2.7 or 3.5 (in a virtual environment). See INSTALL for instruction.

The figures of the paper can be made using the script below.

See the INSTALL instructions

# Run from logistic-bioem directory with 

"run_all.main(run=True,outpath='path_to_save/')"

cd logistic-bioem;

ipython
```
from logistic import run_all
run_all.main(run=True,outpath='paper/')
```

# Run as script from logistic-bioem directory:

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
To rerun the figures without running the Baeysian code:
```
python logistic/run_all.py paper/ --read
```
This should create the macro file and the 5 figures.

# make paper
```
cd paper
pdflatex bfield_bioem; bibtex bfield_bioem; pdflatex bfield_bioem; pdflatex bfield_bioem
```

# Bugs & issues

Please send bug report and/or issues with installation to nicolas.bouche@univ-lyon1.fr
