
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = plotftf1.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

plotftf1:  plotftf1.o 
	$(F77) $(LFLAGS) -o plotftf1 $(OBJECTS) $(PGPLOT) $(BIWT)

plotftf1.o:  plotftf1.f
	$(F77) -c $(FFLAGS) plotftf1.f
