
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = imarsp2.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pglib/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

imarsp2:  imarsp2.o 
	$(F77) $(LFLAGS) -o imarsp2 $(OBJECTS) $(FITSIO) $(BIWT) $(QUEST)

imarsp2.o:  imarsp2.f
	$(F77) -c $(FFLAGS) imarsp2.f
