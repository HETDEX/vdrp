
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = getflim.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pglib/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

getflim:  getflim.o 
	$(F77) $(LFLAGS) -o getflim $(OBJECTS) $(BIWT)

getflim.o:  getflim.f
	$(F77) -c $(FFLAGS) getflim.f
