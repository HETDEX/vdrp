
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = mkimage3d.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pglib/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

mkimage3d:  mkimage3d.o 
	$(F77) $(LFLAGS) -o mkimage3d $(OBJECTS) $(FITSIO) $(BIWT) $(QUEST)

mkimage3d.o:  mkimage3d.f
	$(F77) -c $(FFLAGS) mkimage3d.f
