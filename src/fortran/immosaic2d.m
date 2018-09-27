
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = immosaic2d.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pglib/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

immosaic2d:  immosaic2d.o 
	$(F77) $(LFLAGS) -o immosaic2d $(OBJECTS) $(FITSIO) $(BIWT) $(QUEST)

immosaic2d.o:  immosaic2d.f
	$(F77) -c $(FFLAGS) immosaic2d.f
