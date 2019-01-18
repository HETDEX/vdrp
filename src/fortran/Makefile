F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

BINDIR = $(HOME)/bin
PKG_BINDIR = ../vdrp/bin/

LIBDIR = $(shell vdrp_support_libdir)
HOSTLIBS= -L$(LIBDIR) -lm -lnumrec -lquest -lgcv -llinpack -lpgplot \
	-lcfitsio -L/usr/X11R6/lib -lX11

BINARIES = ampnorm combsed fit2d fit2dc fit2dfix \
	fit2dfw fit2dpsf fitem fitema fitemfull fitemmc fitems fitonesig \
	fitonesiga fitonev fitoneva fitonevp fitonevs fitpoly ftoa ftoaq getflim \
	getflim2 getfwhm getnormexp getsignif getsdssg ima imaddem \
	imaddemn imadderr imar imarfib imarftf imars imarsp imarsp2 \
	imarsp3 imarsqrt imarsvlt imbin imbox2 imbox2sig imboxrow imcheckv imcmb \
	imcmbs imconv imdao imdump imext3d imextnm imextnmq imextsp imextsp2d \
	imextspsig imextspsky imflip imflip2 imhist immask immaskcol imminmax \
	immodfb immosaic2c immosaic2d immosaicv improc imrebin imrep2 imrepe \
	imrepftf imreplace imrepm imrepskyres imrepwv imrot imsect imshift2 \
	imsmline imstat imsum imtwifib imzerofib iso mk2dsp mkimage mkimage3d \
	mkranem plotadc plotbias plotfall plotfall183 \
	plotfall183b plotfl plotftf plotftf1 plotfwhm plotgn3 plotgp plotherr3 \
	plotherr3b plothist plothista plothist2 plothistamp plothistampa \
	plothistcos plothistnoise plothistsky plothisttp plotillum plotline \
	plotlines plotlines1 plotlines2 plotlines2b plotlinesamp plotlinesamp0 \
	plotlinesampa plotlinesc plotlinesoff plotlineswd plotlspres plotlspres2 \
	plotnoise plotplan plotplan2 plots6 plotseda plotsky plotsky2 plotskyres \
	plotsml plotsp18 plotsp2d plotspec1 plotspec2 plotsplines plotsplines0 \
	plotsplinesa plotspring plotspring2 plotstand1 plotstat plotsteve \
	plotsubstat plottimetoc plottp plottrace plotverr plotxy plotxy4 skyftf \
	smftf smfwhm smline smline3 smlineaa smlineftf smlinesp smsp smsp2 smsp3 \
	smsp3twi smspsky smsptwi specclean specint sum2d sum2dsp sumfln \
	sumlineserr sumlineserr_t sumlineserrf sumsign sumspec sumspec100 \
	sumspec2 sumspf sumsplines sumsplinesn sumtmp

BINARIES_NOBIWT = 2dsm atof biwt comp comp0 findpeaks getnorm getoff getoff2 imfixwv mkraster spnorm wvoff

EXTRA_BINARIES = fitlov imfit1d imfit1dp npdyn pfitlov

# SOURCES = $(patsubst %,%.f,$(BINARIES))

# Object files
%.o: %.f Makefile
	$(F77) $(FFLAGS) -c $< -o $@

$(BINARIES_NOBIWT): %: %.o
	$(F77) $< -o $@ $(HOSTLIBS)

$(BINARIES): %: %.o biwgt.o
	$(F77) $< -o $@ biwgt.o $(HOSTLIBS)

fitlov: fitlov.o bconfkg2.o subimsl.o subfitlov.o
	$(F77) $< -o $@ bconfkg2.o subimsl.o subfitlov.o $(HOSTLIBS)

imfit1d: imfit1d.o bconfkg2.o biwgt.o subimsl.o subfitlov.o smooth.o
	$(F77) $< -o $@ bconfkg2.o biwgt.o subimsl.o subfitlov.o smooth.o $(HOSTLIBS)

imfit1dp: imfit1dp.o bconfkg2.o biwgt.o subimsl.o subfitlov.o smooth.o
	$(F77) $< -o $@ bconfkg2.o biwgt.o subimsl.o subfitlov.o smooth.o $(HOSTLIBS)

npdyn: npdyn.o subnpdyn.o
	$(F77) $< -o $@ subnpdyn.o $(HOSTLIBS)

pfitlov: pfitlov.o fitherm.o
	$(F77) $< -o $@ fitherm.o $(HOSTLIBS)

all: $(BINARIES) $(BINARIES_NOBIWT) $(EXTRA_BINARIES)

bins: 
	@echo $(BINARIES)

.phony:
clean:
	-rm -f $(BINARIES) $(BINARIES_NOBIWT) $(EXTRA_BINARIES) *.o

.phony:
pkg_clean: clean
	-rm -r $(addprefix $(PKG_BINDIR)/,$(BINARIES))
	-rm -r $(addprefix $(PKG_BINDIR)/,$(BINARIES_NOBIWT))
	-rm -r $(addprefix $(PKG_BINDIR)/,$(EXTRA_BINARIES))

.phony:
install: all
	mkdir -p $(BINDIR)
	cp $(BINARIES) $(BINARIES_NOBIWT) $(EXTRA_BINARIES) $(BINDIR)/

.phony:
pkg_install: all
	mkdir -p $(PKG_BINDIR)
	cp $(BINARIES) $(BINARIES_NOBIWT) $(EXTRA_BINARIES) $(PKG_BINDIR)/
