
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = imfit1dp.o bconfkg2.o subimsl.o subfitlov.o smooth.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pgplot/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

imfit1dp:  imfit1dp.o 
	$(F77) $(LFLAGS) -o imfit1dp $(OBJECTS) $(FITSIO) $(QUEST) $(BIWT) $(NUMREC) $(GCV) $(PGPLOT)

imfit1dp.o:  imfit1dp.f
	$(F77) -c $(FFLAGS) imfit1dp.f
