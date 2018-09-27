
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = imext3d.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pglib/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

imext3d:  imext3d.o 
	$(F77) $(LFLAGS) -o imext3d $(OBJECTS) $(FITSIO) $(BIWT) $(QUEST)

imext3d.o:  imext3d.f
	$(F77) -c $(FFLAGS) imext3d.f
