c	Karl's front-end program for adker.
c
	parameter(nmax=30000,igrid=100,isimmax=300)
	real x(nmax),arrn(igrid),xout(nmax),xin(isimmax),yin(10)
        real xsim(isimmax,igrid),x05(igrid),x95(igrid)
        real xl(igrid),xnew(nmax),arrn2(igrid),xl2(igrid)
	character file1*40

	parameter(zero=0.e0,one=1.e0,big=1.e10)

	itype=1
        ibest=0

1069	format(a)

1	call qc1('Input data ','ncont1.def',file1)
	open(unit=1,file=file1,status='old',err=1)
	
10	call qi1('Input grid size ','ncont1.def',igin)
	if(igin.gt.100) then
	   write(*,"('100 is max, try again')")
	   goto 10
	endif
        ain=0
	call qr1('Window width (in bins) ','ncont1.def',hin)
        gin=0
	call qr2('What are the min and max values ','ncont1.def',
     $       xmin,xmax)
c        call qr2('Radii range ','ncont1.def',rmin,rmax)

c        call qr2('What the mean and vel. disp. ','ncont1.def',
c     $       vmean,vdisp)

c        call qi1('Double (2) or nothing (1) ','ncont1.def',ido)
        call savdef

	n=0
        xmin1=big
        xmax1=-big
	x2=0
	do i=1,nmax
	   read(1,*,end=666) x1
	   n=n+1
	   x(n)=x1
	   xmin1=min(xmin1,x(n))
	   xmax1=max(xmax1,x(n))
	enddo
        write(*,1002) 'Too many points, nmax is only : ',nmax
 1002   format(a32,i7)
 666    continue

        if (xmin.eq.xmax) then
           xmin=xmin1
           xmax=xmax1
        endif

	write(*,1001) 'There are ',n,'points'
 1001   format(a10,i6,1x,a6)

	call pgbegin(0,'?',1,1)
	call pgpap(0.,1.)
	call pgscf(2)
	call pgsch(1.4)
	call pgslw(2)

 668    continue

        ip=1
        xminin=xmin
        xmaxin=xmax

	call adker1(x,n,itype,igin,hin,ain,gin,ip,arrn,
     $       xminin,xmaxin,xout,xl)

        if(ibest.eq.1) goto 667

        call qi1('Find best smoothing (1-yes) ','ncont1.def',ians)
        if(ians.ne.1) goto 667

        call qr2('Start at what h and step ','ncont1.def',hmin,step)
        call savdef

        in=0
	i1=0
        ymin=big
        ymax=-big
        do hin=hmin,hmin+10.*step,step
           call adkerk2(xout,n,igin,hin,ans)
           if(i1.ne.1) then
              print *,hin,ans/abs(ans)
              h1=abs(ans)
              i1=1
           else
              print *,hin,ans/h1
           endif
           in=in+1
           xin(in)=hin
           yin(in)=ans
           ymin=min(ymin,yin(in))
           ymax=max(ymax,yin(in))
        enddo

        call pgenv(hmin,hmin+10.*step,ymin,ymax,0,0)
        call pgsch(1.5)
        call pgslw(1)
        call pgpoint(in,xin,yin,17)
        call pgsch(1.)

        write(*,"('Again (1-yes) : '$)")
        read *,ians
        if(ians.ne.1) goto 667
        write(*,"('Best h : '$)")
        read *,hin
        ibest=1
        goto 668

 667    continue

        call qi1('Bootstrap errors (1-yes) ','ncont1.def',ians2)
	call savdef
        if(ians2.ne.1) goto 669

        idum=-1
        do isim=1,isimmax
           do i=1,n
              iran=nint(ran2(idum)*(n-1))+1
              xnew(i)=x(iran)
 670          continue
           enddo

           ip=0
           xminin=xmin
           xmaxin=xmax

           call adker1(xnew,n,itype,igin,hin,ain,gin,ip,arrn2,
     $          xminin,xmaxin,xout,xl2)
           
           write(*,'(a8,i4,1x,f6.4,a1,$)')
     $          'Did sim ',isim,arrn2(igin/2),char(13)
	   call flush(6)

           do i=1,igin
              xsim(isim,i)=arrn2(i)
           enddo

        enddo

        do i=1,igin
           do is=1,isimmax
              xin(is)=xsim(is,i)
           enddo
           call biwgt(xin,isimmax,xb,xs)
           i5=nint(float(is)*.05)
           i5=max(1,i5)
           i95=nint(float(is)*.95)
           i95=min(isimmax,i95)
           i16=nint(float(is)*.16)
           i16=max(1,i16)
           i84=nint(float(is)*.84)
           i84=min(isimmax,i84)
           x05(i)=xin(i16)
           x95(i)=xin(i84)
        enddo

 669    continue

        open(unit=11,file='adker1.out',status='unknown')
        do i=1,igin
           if(ians2.eq.1) then
              write(11,1101) xl(i),arrn(i),x05(i),x95(i)
           else
              write(11,*) xl(i),arrn(i)
           endif
        enddo
        close(11)
 1101	format(f8.2,3(1x,f10.5))

        if(ians2.eq.1) then
           call pgsls(2)
           call pgline(igin,xl,x05)
           call pgline(igin,xl,x95)
        endif

	sig=1.
	gnorm=1.
	do i=1,igin
	   xexp=xl(i)**2/2./sig/sig
	   x05(i)=gnorm/sqrt(2.*3.14*sig*sig)*exp(-xexp)
	enddo
	call pgsls(1)
c	call pgline(igin,xl,x05)

        call pgend

	stop
	end

      FUNCTION ran2(idum)
      INTEGER idum,IM1,IM2,IMM1,IA1,IA2,IQ1,IQ2,IR1,IR2,NTAB,NDIV
      REAL ran2,AM,EPS,RNMX
      PARAMETER (IM1=2147483563,IM2=2147483399,AM=1./IM1,IMM1=IM1-1,
     *IA1=40014,IA2=40692,IQ1=53668,IQ2=52774,IR1=12211,IR2=3791,
     *NTAB=32,NDIV=1+IMM1/NTAB,EPS=1.2e-7,RNMX=1.-EPS)
      INTEGER idum2,j,k,iv(NTAB),iy
      SAVE iv,iy,idum2
      DATA idum2/123456789/, iv/NTAB*0/, iy/0/
      if (idum.le.0) then
        idum=max(-idum,1)
        idum2=idum
        do 11 j=NTAB+8,1,-1
          k=idum/IQ1
          idum=IA1*(idum-k*IQ1)-k*IR1
          if (idum.lt.0) idum=idum+IM1
          if (j.le.NTAB) iv(j)=idum
11      continue
        iy=iv(1)
      endif
      k=idum/IQ1
      idum=IA1*(idum-k*IQ1)-k*IR1
      if (idum.lt.0) idum=idum+IM1
      k=idum2/IQ2
      idum2=IA2*(idum2-k*IQ2)-k*IR2
      if (idum2.lt.0) idum2=idum2+IM2
      j=1+iy/NDIV
      iy=iv(j)-idum2
      iv(j)=idum
      if(iy.lt.1)iy=iy+IMM1
      ran2=min(AM*iy,RNMX)
      return
      END
