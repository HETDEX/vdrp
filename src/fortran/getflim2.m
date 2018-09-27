
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = getflim2.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pglib/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

getflim2:  getflim2.o 
	$(F77) $(LFLAGS) -o getflim2 $(OBJECTS) $(BIWT)

getflim2.o:  getflim2.f
	$(F77) -c $(FFLAGS) getflim2.f
