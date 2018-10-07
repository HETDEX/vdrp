
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = plotlines2b.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

plotlines2b:  plotlines2b.o 
	$(F77) $(LFLAGS) -o plotlines2b $(OBJECTS) $(PGPLOT) $(BIWT)

plotlines2b.o:  plotlines2b.f
	$(F77) -c $(FFLAGS) plotlines2b.f
