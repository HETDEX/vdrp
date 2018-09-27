
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = pfitlov.o fitherm.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

pfitlov:  pfitlov.o 
	$(F77) $(LFLAGS) -o pfitlov $(OBJECTS) $(FITSIO) $(QUEST) $(NUMREC) $(PGPLOT)

pfitlov.o:  pfitlov.f
	$(F77) -c $(FFLAGS) pfitlov.f
