
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = plotlines1.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

plotlines1:  plotlines1.o 
	$(F77) $(LFLAGS) -o plotlines1 $(OBJECTS) $(PGPLOT) $(BIWT)

plotlines1.o:  plotlines1.f
	$(F77) -c $(FFLAGS) plotlines1.f
