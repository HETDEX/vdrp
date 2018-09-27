
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = fit2dpsf.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

fit2dpsf:  fit2dpsf.o 
	$(F77) $(LFLAGS) -o fit2dpsf $(OBJECTS) $(FITSIO) $(BIWT) $(QUEST) $(PGPLOT)

fit2dpsf.o:  fit2dpsf.f
	$(F77) -c $(FFLAGS) fit2dpsf.f
