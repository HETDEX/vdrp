
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = plotnoise.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

plotnoise:  plotnoise.o 
	$(F77) $(LFLAGS) -o plotnoise $(OBJECTS) $(PGPLOT) $(BIWT)

plotnoise.o:  plotnoise.f
	$(F77) -c $(FFLAGS) plotnoise.f
