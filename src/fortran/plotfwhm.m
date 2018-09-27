
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = plotfwhm.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

plotfwhm:  plotfwhm.o 
	$(F77) $(LFLAGS) -o plotfwhm $(OBJECTS) $(PGPLOT) $(BIWT)

plotfwhm.o:  plotfwhm.f
	$(F77) -c $(FFLAGS) plotfwhm.f
