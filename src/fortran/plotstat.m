
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = plotstat.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

plotstat:  plotstat.o 
	$(F77) $(LFLAGS) -o plotstat $(OBJECTS) $(PGPLOT) $(BIWT) $(QUEST) $(NUMREC)

plotstat.o:  plotstat.f
	$(F77) -c $(FFLAGS) plotstat.f
