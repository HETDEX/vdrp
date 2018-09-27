
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = plotstand1.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

plotstand1:  plotstand1.o 
	$(F77) $(LFLAGS) -o plotstand1 $(OBJECTS) $(PGPLOT) $(BIWT) $(GCV)

plotstand1.o:  plotstand1.f
	$(F77) -c $(FFLAGS) plotstand1.f
