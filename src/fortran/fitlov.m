
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = fitlov.o bconfkg2.o subimsl.o subfitlov.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

fitlov:  fitlov.o 
	$(F77) $(LFLAGS) -o fitlov $(OBJECTS) $(FITSIO) $(QUEST) $(NUMREC) $(PGPLOT)

fitlov.o:  fitlov.f
	$(F77) -c $(FFLAGS) fitlov.f
