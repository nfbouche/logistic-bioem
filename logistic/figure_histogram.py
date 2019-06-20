
import sys 

def run(table1,table2,outpdf):
    from Modelling import Histogram_Bfield
    hist=Histogram_Bfield(humans=table1,rats=table2)
    hist.make_plot(outpdf=outpdf,verbose=True)
    print("Histogram: P-val",hist.Pval)


if __name__ == '__main__':
	run(table1=sys.argv[1],table2=sys.argv[2],outpdf=sys.argv[3])
