
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = 2dsm.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvlib.a
LINPACK=  ~gebhardt/lib/linpack/linpack.a
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

2dsm:  2dsm.o 
	$(F77) $(LFLAGS) -o 2dsm $(OBJECTS) $(PGPLOT) $(QUEST) $(GCV) $(LINPACK)

2dsm.o:  2dsm.f
	$(F77) -c $(FFLAGS) 2dsm.f
