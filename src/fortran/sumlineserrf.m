
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = sumlineserrf.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

sumlineserrf:  sumlineserrf.o 
	$(F77) $(LFLAGS) -o sumlineserrf $(OBJECTS) $(PGPLOT) $(BIWT)

sumlineserrf.o:  sumlineserrf.f
	$(F77) -c $(FFLAGS) sumlineserrf.f
