
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = imhist.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

imhist:  imhist.o 
	$(F77) $(LFLAGS) -o imhist $(OBJECTS) $(FITSIO) $(BIWT) $(QUEST) $(PGPLOT)

imhist.o:  imhist.f
	$(F77) -c $(FFLAGS) imhist.f
