
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = imflip2.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pglib/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

imflip2:  imflip2.o 
	$(F77) $(LFLAGS) -o imflip2 $(OBJECTS) $(FITSIO) $(BIWT) $(QUEST)

imflip2.o:  imflip2.f
	$(F77) -c $(FFLAGS) imflip2.f
