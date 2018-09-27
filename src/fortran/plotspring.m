
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = plotspring.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

plotspring:  plotspring.o 
	$(F77) $(LFLAGS) -o plotspring $(OBJECTS) $(PGPLOT) $(BIWT)

plotspring.o:  plotspring.f
	$(F77) -c $(FFLAGS) plotspring.f
