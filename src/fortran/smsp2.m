
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = smsp2.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pglib/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

smsp2:  smsp2.o 
	$(F77) $(LFLAGS) -o smsp2 $(OBJECTS) $(FITSIO) $(BIWT) $(QUEST)

smsp2.o:  smsp2.f
	$(F77) -c $(FFLAGS) smsp2.f
