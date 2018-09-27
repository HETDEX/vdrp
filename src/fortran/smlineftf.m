
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = smlineftf.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

smlineftf:  smlineftf.o 
	$(F77) $(LFLAGS) -o smlineftf $(OBJECTS) $(PGPLOT) $(BIWT) $(QUEST) $(GCV)

smlineftf.o:  smlineftf.f
	$(F77) -c $(FFLAGS) smlineftf.f
