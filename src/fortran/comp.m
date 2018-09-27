
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = comp.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pglib/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

comp:  comp.o 
	$(F77) $(LFLAGS) -o comp $(OBJECTS)

comp.o:  comp.f
	$(F77) -c $(FFLAGS) comp.f
