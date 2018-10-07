
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = plotlines2.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

plotlines2:  plotlines2.o 
	$(F77) $(LFLAGS) -o plotlines2 $(OBJECTS) $(PGPLOT) $(BIWT)

plotlines2.o:  plotlines2.f
	$(F77) -c $(FFLAGS) plotlines2.f
