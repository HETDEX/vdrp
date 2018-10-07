
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = fitemfull.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

fitemfull:  fitemfull.o 
	$(F77) $(LFLAGS) -o fitemfull $(OBJECTS) $(BIWT) $(QUEST) $(PGPLOT)

fitemfull.o:  fitemfull.f
	$(F77) -c $(FFLAGS) fitemfull.f
