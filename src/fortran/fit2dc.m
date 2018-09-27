
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = fit2dc.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

fit2dc:  fit2dc.o 
	$(F77) $(LFLAGS) -o fit2dc $(OBJECTS) $(FITSIO) $(BIWT) $(QUEST) $(PGPLOT)

fit2dc.o:  fit2dc.f
	$(F77) -c $(FFLAGS) fit2dc.f
