
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = plotsp2d.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

plotsp2d:  plotsp2d.o 
	$(F77) $(LFLAGS) -o plotsp2d $(OBJECTS) $(PGPLOT) $(BIWT)

plotsp2d.o:  plotsp2d.f
	$(F77) -c $(FFLAGS) plotsp2d.f
