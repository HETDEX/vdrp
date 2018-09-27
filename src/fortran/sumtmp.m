
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = sumtmp.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

sumtmp:  sumtmp.o 
	$(F77) $(LFLAGS) -o sumtmp $(OBJECTS) $(PGPLOT) $(BIWT)

sumtmp.o:  sumtmp.f
	$(F77) -c $(FFLAGS) sumtmp.f
