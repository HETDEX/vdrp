
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = plotgn3.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

plotgn3:  plotgn3.o 
	$(F77) $(LFLAGS) -o plotgn3 $(OBJECTS) $(PGPLOT) $(BIWT)

plotgn3.o:  plotgn3.f
	$(F77) -c $(FFLAGS) plotgn3.f
