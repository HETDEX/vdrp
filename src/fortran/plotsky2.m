
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = plotsky2.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

plotsky2:  plotsky2.o 
	$(F77) $(LFLAGS) -o plotsky2 $(OBJECTS) $(PGPLOT) $(BIWT)

plotsky2.o:  plotsky2.f
	$(F77) -c $(FFLAGS) plotsky2.f
