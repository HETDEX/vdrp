
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = smlineaa.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

smlineaa:  smlineaa.o 
	$(F77) $(LFLAGS) -o smlineaa $(OBJECTS) $(PGPLOT) $(BIWT) $(QUEST) $(GCV)

smlineaa.o:  smlineaa.f
	$(F77) -c $(FFLAGS) smlineaa.f
