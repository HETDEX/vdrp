
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = plotspec2.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

plotspec2:  plotspec2.o 
	$(F77) $(LFLAGS) -o plotspec2 $(OBJECTS) $(PGPLOT) $(BIWT)

plotspec2.o:  plotspec2.f
	$(F77) -c $(FFLAGS) plotspec2.f
