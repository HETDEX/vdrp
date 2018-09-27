
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = smline3.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

smline3:  smline3.o 
	$(F77) $(LFLAGS) -o smline3 $(OBJECTS) $(PGPLOT) $(BIWT) $(QUEST) $(GCV)

smline3.o:  smline3.f
	$(F77) -c $(FFLAGS) smline3.f
