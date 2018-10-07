
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = fit2d.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

fit2d:  fit2d.o 
	$(F77) $(LFLAGS) -o fit2d $(OBJECTS) $(FITSIO) $(BIWT) $(QUEST) $(PGPLOT)

fit2d.o:  fit2d.f
	$(F77) -c $(FFLAGS) fit2d.f
