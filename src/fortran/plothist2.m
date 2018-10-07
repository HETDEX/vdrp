
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = plothist2.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

plothist2:  plothist2.o 
	$(F77) $(LFLAGS) -o plothist2 $(OBJECTS) $(PGPLOT) $(BIWT)

plothist2.o:  plothist2.f
	$(F77) -c $(FFLAGS) plothist2.f
