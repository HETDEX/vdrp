
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = sumlineserr.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

sumlineserr:  sumlineserr.o 
	$(F77) $(LFLAGS) -o sumlineserr $(OBJECTS) $(PGPLOT) $(BIWT)

sumlineserr.o:  sumlineserr.f
	$(F77) -c $(FFLAGS) sumlineserr.f
