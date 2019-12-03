F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

OS := $(shell uname)

ifeq ($(READTHEDOCS),True)
	PKG_INSTALL_TARGETS =
else
	PKG_INSTALL_TARGETS = all install_pkg
endif

ifeq ($(OS),Darwin)
	FFLAGS = -O3 -I/usr/X11/include
	X11LIBS = -L/usr/X11/lib
	HOSTLIBS= -L$(LIBDIR) -lm -lnumrec -lquest -lgcv -llinpack -lpgplot \
		-lcfitsio $(X11LIBS) -lX11
else
	X11LIBS = -L/usr/X11R6/lib
	HOSTLIBS= -L$(LIBDIR) -lm -lnumrec -lquest -lgcv -llinpack -lpgplot \
		-lcfitsio $(X11LIBS) -lX11
endif

BINDIR = $(HOME)/bin
PKG_BINDIR = ../vdrp/bin/

LIBDIR = $(shell vdrp_support_libdir)

BINARIES = combsed fit2d fitem fitonevp fitradecsp getnormexp getsdssg \
	imextsp imrot immosaicv mkimage mkimage3d plotseda specclean \
    sumlineserr sumspec sumsplines

BINARIES_NOBIWT = biwt getoff2

# SOURCES = $(patsubst %,%.f,$(BINARIES))

# Object files
%.o: %.f Makefile
	$(F77) $(FFLAGS) -c $< -o $@

$(BINARIES_NOBIWT): %: %.o
	$(F77) $< -o $@ $(HOSTLIBS)

$(BINARIES): %: %.o biwgt.o
	$(F77) $< -o $@ biwgt.o $(HOSTLIBS)

all: $(BINARIES) $(BINARIES_NOBIWT)

bins:
	@echo $(BINARIES)

.phony:
clean:
	-rm -f $(BINARIES) $(BINARIES_NOBIWT) *.o

.phony:
pkg_clean: clean
	-rm -r $(addprefix $(PKG_BINDIR)/,$(BINARIES))
	-rm -r $(addprefix $(PKG_BINDIR)/,$(BINARIES_NOBIWT))

.phony:
install: all
	mkdir -p $(BINDIR)
	cp $(BINARIES) $(BINARIES_NOBIWT) $(BINDIR)/

pkg_install: $(PKG_INSTALL_TARGETS)

install_pkg:
	$(info ${OS})
	mkdir -p $(PKG_BINDIR)
	cp $(BINARIES) $(BINARIES_NOBIWT) $(PKG_BINDIR)/
