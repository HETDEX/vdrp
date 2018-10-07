
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = plotlineswd.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

plotlineswd:  plotlineswd.o 
	$(F77) $(LFLAGS) -o plotlineswd $(OBJECTS) $(PGPLOT) $(BIWT)

plotlineswd.o:  plotlineswd.f
	$(F77) -c $(FFLAGS) plotlineswd.f
