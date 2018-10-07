
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = plotsp18.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

plotsp18:  plotsp18.o 
	$(F77) $(LFLAGS) -o plotsp18 $(OBJECTS) $(PGPLOT) $(BIWT)

plotsp18.o:  plotsp18.f
	$(F77) -c $(FFLAGS) plotsp18.f
