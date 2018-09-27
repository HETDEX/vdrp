
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = imbox2sig.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pglib/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

imbox2sig:  imbox2sig.o 
	$(F77) $(LFLAGS) -o imbox2sig $(OBJECTS) $(FITSIO) $(BIWT) $(QUEST)

imbox2sig.o:  imbox2sig.f
	$(F77) -c $(FFLAGS) imbox2sig.f
