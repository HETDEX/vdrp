
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = imaddem.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pglib/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

imaddem:  imaddem.o 
	$(F77) $(LFLAGS) -o imaddem $(OBJECTS) $(FITSIO) $(BIWT) $(QUEST)

imaddem.o:  imaddem.f
	$(F77) -c $(FFLAGS) imaddem.f
