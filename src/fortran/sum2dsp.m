
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = sum2dsp.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11

sum2dsp:  sum2dsp.o 
	$(F77) $(LFLAGS) -o sum2dsp $(OBJECTS) $(PGPLOT) $(BIWT)

sum2dsp.o:  sum2dsp.f
	$(F77) -c $(FFLAGS) sum2dsp.f
