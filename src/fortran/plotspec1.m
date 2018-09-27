
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = plotspec1.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

plotspec1:  plotspec1.o 
	$(F77) $(LFLAGS) -o plotspec1 $(OBJECTS) $(PGPLOT) $(BIWT)

plotspec1.o:  plotspec1.f
	$(F77) -c $(FFLAGS) plotspec1.f
