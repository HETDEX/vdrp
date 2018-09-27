
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = ima.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

ima:  ima.o 
	$(F77) $(LFLAGS) -o ima $(OBJECTS) $(FITSIO) $(BIWT) $(QUEST) $(PGPLOT) $(GCV)

ima.o:  ima.f
	$(F77) -c $(FFLAGS) ima.f
