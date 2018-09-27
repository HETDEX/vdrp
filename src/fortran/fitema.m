
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = fitema.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

fitema:  fitema.o 
	$(F77) $(LFLAGS) -o fitema $(OBJECTS) $(BIWT) $(QUEST) $(PGPLOT)

fitema.o:  fitema.f
	$(F77) -c $(FFLAGS) fitema.f
