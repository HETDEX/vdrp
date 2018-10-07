
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = fit2dfw.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

fit2dfw:  fit2dfw.o 
	$(F77) $(LFLAGS) -o fit2dfw $(OBJECTS) $(FITSIO) $(BIWT) $(QUEST) $(PGPLOT)

fit2dfw.o:  fit2dfw.f
	$(F77) -c $(FFLAGS) fit2dfw.f
