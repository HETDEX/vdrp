
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = imrep2.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pglib/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

imrep2:  imrep2.o 
	$(F77) $(LFLAGS) -o imrep2 $(OBJECTS) $(FITSIO) $(BIWT)

imrep2.o:  imrep2.f
	$(F77) -c $(FFLAGS) imrep2.f
