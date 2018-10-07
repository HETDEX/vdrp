
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = immodfb.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pglib/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

immodfb:  immodfb.o 
	$(F77) $(LFLAGS) -o immodfb $(OBJECTS) $(FITSIO) $(BIWT) $(QUEST)

immodfb.o:  immodfb.f
	$(F77) -c $(FFLAGS) immodfb.f
