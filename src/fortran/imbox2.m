
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = imbox2.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pglib/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

imbox2:  imbox2.o 
	$(F77) $(LFLAGS) -o imbox2 $(OBJECTS) $(FITSIO) $(BIWT) $(QUEST)

imbox2.o:  imbox2.f
	$(F77) -c $(FFLAGS) imbox2.f
