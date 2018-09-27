
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = plothistcos.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

plothistcos:  plothistcos.o 
	$(F77) $(LFLAGS) -o plothistcos $(OBJECTS) $(PGPLOT) $(BIWT)

plothistcos.o:  plothistcos.f
	$(F77) -c $(FFLAGS) plothistcos.f
