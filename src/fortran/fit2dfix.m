
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = fit2dfix.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

fit2dfix:  fit2dfix.o 
	$(F77) $(LFLAGS) -o fit2dfix $(OBJECTS) $(FITSIO) $(BIWT) $(QUEST) $(PGPLOT)

fit2dfix.o:  fit2dfix.f
	$(F77) -c $(FFLAGS) fit2dfix.f
