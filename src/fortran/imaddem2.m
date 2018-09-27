
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = imaddem2.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pglib/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

imaddem2:  imaddem2.o 
	$(F77) $(LFLAGS) -o imaddem2 $(OBJECTS) $(FITSIO) $(BIWT) $(QUEST)

imaddem2.o:  imaddem2.f
	$(F77) -c $(FFLAGS) imaddem2.f
