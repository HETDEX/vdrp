
F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OBJECTS = mk2dsp.o
HOSTLIBS= -lm
NUMREC =  ~gebhardt/lib/numrec/numrec.a
BIWT   =  ~gebhardt/progs/biwgt.o
QUEST  =  ~gebhardt/lib/libquest/libquest.o
GCV    =  ~gebhardt/lib/gcv/gcvspl.o
PGPLOT =  ~gebhardt/lib/pglib/libpgplot.a -L/usr/X11R6/lib -lX11
FITSIO =  ~gebhardt/lib/cfitsio/libcfitsio.a

mk2dsp:  mk2dsp.o 
	$(F77) $(LFLAGS) -o mk2dsp $(OBJECTS) $(FITSIO) $(BIWT) $(QUEST)

mk2dsp.o:  mk2dsp.f
	$(F77) -c $(FFLAGS) mk2dsp.f
