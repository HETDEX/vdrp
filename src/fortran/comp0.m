
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = comp0.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pglib/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

comp0:  comp0.o 
	$(F77) $(LFLAGS) -o comp0 $(OBJECTS)

comp0.o:  comp0.f
	$(F77) -c $(FFLAGS) comp0.f
