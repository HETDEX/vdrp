
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = plotlspres2.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

plotlspres2:  plotlspres2.o 
	$(F77) $(LFLAGS) -o plotlspres2 $(OBJECTS) $(PGPLOT) $(BIWT)

plotlspres2.o:  plotlspres2.f
	$(F77) -c $(FFLAGS) plotlspres2.f
