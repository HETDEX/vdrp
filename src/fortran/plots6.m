
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = plots6.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

plots6:  plots6.o 
	$(F77) $(LFLAGS) -o plots6 $(OBJECTS) $(PGPLOT) $(BIWT)

plots6.o:  plots6.f
	$(F77) -c $(FFLAGS) plots6.f
