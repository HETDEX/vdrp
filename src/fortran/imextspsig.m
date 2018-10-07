
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = imextspsig.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pglib/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

imextspsig:  imextspsig.o 
	$(F77) $(LFLAGS) -o imextspsig $(OBJECTS) $(FITSIO) $(BIWT) $(QUEST)

imextspsig.o:  imextspsig.f
	$(F77) -c $(FFLAGS) imextspsig.f
