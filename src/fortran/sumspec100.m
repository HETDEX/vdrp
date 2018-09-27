
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = sumspec100.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pglib/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

sumspec100:  sumspec100.o 
	$(F77) $(LFLAGS) -o sumspec100 $(OBJECTS) $(BIWT)

sumspec100.o:  sumspec100.f
	$(F77) -c $(FFLAGS) sumspec100.f
