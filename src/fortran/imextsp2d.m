
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = imextsp2d.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pglib/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

imextsp2d:  imextsp2d.o 
	$(F77) $(LFLAGS) -o imextsp2d $(OBJECTS) $(FITSIO) $(BIWT) $(QUEST)

imextsp2d.o:  imextsp2d.f
	$(F77) -c $(FFLAGS) imextsp2d.f
