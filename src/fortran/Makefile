F77=gfortran
FFLAGS = -O3
LFLAGS = -O3

LIBDIR = $(shell hetdex_vdrp_libdir)
HOSTLIBS= -L$(LIBDIR) -lm -Lnumrec -Lquest -Lgcv -Llinpack -lpgplot.a \
	-lcfitsio-L/usr/X11R6/lib -lX11

BINARIES = 2dsm ampnorm atof combsed comp comp0 findpeaks fit2d fit2dc fit2dfix \
	fit2dfw fit2dpsf fitem fitema fitemfull fitemmc fitems fitlov fitonesig \
	fitonesiga fitonev fitoneva fitonevp fitonevs fitpoly ftoa ftoaq getflim \
	getflim2 getfwhm getnorm getsignif ima imaddem imaddem2 imaddemfw imaddemn \
	imadderr imar imarfib imarftf imars imarsp imarsp2 imarsp3 imarsqrt imarsvlt \
	imbin imbox2 imbox2sig imboxrow imcheckv imcmb imcmbs imconv imdao imdump \
	imext3d imextnm imextnmq imextsp imextsp2d imextspsig imextspsky imfit1d \
	imfit1dp imfixwv imflip imflip2 imhist immask immaskcol imminmax immodfb \
	immosaic2c immosaic2d immosaicv improc imrebin imrep2 imrepe imrepftf \
	imreplace imrepm imrepskyres imrepwv imrot imsect imshift2 imsmline \
	imstat imsum imtwifib imzerofib iso mk2dsp mkimage mkimage3d mkranem mkraster \
	pdyn pfitlov plotadc plotbias plotfall plotfall183 plotfall183b plotfl plotftf \
	plotftf1 plotfwhm plotgn3 plotgp plotherr3 plothist plothist2 plothistamp plothistampa \
	plothistcos plothistnoise plothistsky plothisttp plotillum plotline plotlines \
	plotlines1 plotlines2 plotlines2b plotlinesamp plotlinesamp0 plotlinesc	\
	plotlinesoff plotlineswd plotlspres plotlspres2 plotnoise plotplan plots6 \
	plotseda plotsky plotsky2 plotskyres plotsml plotsp18 plotsp2d plotspec1 \
	plotspec2 plotsplines plotsplines0 plotsplinesa plotspring plotspring2 \
	plotstand1 plotstat plotsteve plotsubstat plottimetoc plottp plottrace plotverr plotxy \
	plotxy4 skyftf smftf smfwhm smline smline3 smlineaa smlineftf smlinesp smsp \
	smsp2 smsp3 smsp3twi smspsky smsptwi specclean specint spnorm sum2d sum2dsp sumfln \
	sumlineserr sumlineserrf sumsign sumspec sumspec100 sumspec2 sumspf \
	sumsplines sumsplinesn sumtmp wvoff

SOURCES = $(patsubst %,%.f,$(BINARIES))

# Object files
%.o: %.f Makefile
	$(F77) $(FFLAGS) $< -o $@

$(BINARIES): %: %.o biwgt.o
	$(F77) $< -o $@ $(HOSTLIBS)

