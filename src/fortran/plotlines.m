
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = plotlines.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

plotlines:  plotlines.o 
	$(F77) $(LFLAGS) -o plotlines $(OBJECTS) $(PGPLOT) $(BIWT)

plotlines.o:  plotlines.f
	$(F77) -c $(FFLAGS) plotlines.f
