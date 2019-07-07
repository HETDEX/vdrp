
      parameter(nmax=1000,nsimt=1000,ntemp=30)

      real v(nmax)
      real rad(nmax),vel(nmax),sig(nmax),sigl(nmax),sigh(nmax)
      real vell(nmax),velh(nmax),fsim(nsimt),xsim(nsimt,nmax)
      real yin(nsimt),fw(nmax),vs(nmax),stype1(ntemp,nsimt)
      real stype(nmax,ntemp),erry(nmax)
      real covar(6,6),a(6),h3a(nsimt),h4a(nsimt),h3(nmax),h3h(nmax)
      real h3l(nmax),h4(nmax),h4h(nmax),h4l(nmax)
      integer ia(6)
      character file1*40
      data big/1.e30/

      parameter(pi=3.1415926539)

 1    call qc1('File of Sims ','pallmc.def',file1)
      call savdef
      call pgbegin(0,'?',1,2)
      call pgscf(2)
      call pgsch(1.3)
      call pgslw(2)

      do i=1,5
         ia(i)=1
      enddo
c      ia(4)=0
c      ia(5)=0
      ia(6)=0

      open(unit=1,file=file1,status='old',err=1)

 1001 format(7x,i3)
 1002 format('Sigma Bias : ',4(1x,f8.3))

      open(unit=11,file='pallmc.out',status='unknown')
      open(unit=12,file='v.out',status='unknown')
      open(unit=13,file='pallmc2.out',status='unknown')

      radmin=big
      radmax=-big
      sigmin=big
      sigmax=-big
      velmin=big
      velmax=-big
      nf=0
      do if=1,nmax
         read(1,*,end=666) file1,rsim
         do i=1,40
            if(file1(i:i).eq.' ') then
               nfile=i-1
               goto 966
            endif
         enddo
 966     continue
c         if(if.eq.1) then
            open(unit=2,file=file1(1:nfile-3)//'mcfit',status='old',
     $        err=766)
            goto 768
 766        continue
            close(2)
            print *,"File not Found: ",file1(1:nfile-4)
            goto 767
 768        continue
            read(2,*) nl,ns
            do i=1,nl
               read(2,*) x1,x2,x3,x4,x5,x6
               v(i)=x1
            enddo
            close(2)
c         endif
         nf=nf+1
         open(unit=2,file=file1(1:nfile),status='old')
         read(2,*) nsim,nl,ns
         read(2,*) (fsim(i),i=1,nsim)
         i16=max(1,nint(float(nsim)*.16))
         i84=min(nsim,nint(float(nsim)*.84))
         do i=1,nl+ns+1
            read(2,*) (yin(j),j=1,nsim)
            do j=1,nsim
               xsim(j,i)=yin(j)
            enddo
         enddo
         close(2)
         open(unit=2,file=file1(1:nfile-3)//'mcfit2',status='unknown')
         do i=1,nl
            do j=1,nsim
               yin(j)=xsim(j,i)
            enddo
            call biwgt(yin,nsim,xb,xs)
            bias=xsim(1,i)-xb
            y16=xsim(1,i)-(xb-yin(i16))
            y84=xsim(1,i)+(yin(i84)-xb)
            write(2,2001) v(i),xsim(1,i),y16,y84
c            print *,xs,y84-xsim(1,i),xs/(y84-xsim(1,i))
         enddo
         close(2)
c - get the distribution of fvalues
         do j=1,nsim
            yin(j)=fsim(j)
         enddo
         call biwgt(yin,nsim,xb,xs)
         nj=0
         fcut=fsim(1)+xs+1e10
         do j=1,nsim
            if(fsim(j).le.fcut) then
               nj=nj+1
               do i=1,nl
                  yin(i)=xsim(j,i)
c                  if(i.le.9) yin(i)=0.
                  erry(i)=1.
               enddo
               call getfwhm(nl,v,yin,0.5,fwhm,vmax,ymax,v1,v2)
               a(1)=ymax*sqrt(2.*pi)*fwhm/2.35
               a(2)=vmax
               a(3)=fwhm/2.35
               fwhmo=fwhm
               a(4)=0.
               a(5)=0.
               a(6)=0.
               call fithermec(nl,v,yin,erry,a,ia,6,covar)
               vs(nj)=a(2)
               fw(nj)=2.35*a(3)
c               vs(nj)=v1
c               fw(nj)=2.35*v2
c               fw(nj)=fwhmo
               h3a(nj)=a(4)
               h4a(nj)=a(5)
c               if(a(5).gt.0.01) nj=nj-1
               write(12,*) nj,a(2),a(3),a(4),a(5)
               do i=1,ns
                  stype1(i,nj)=xsim(nj,nl+i)
               enddo
            endif
         enddo
         i16=max(1,nint(float(nj)*.16))
         i84=min(nj,nint(float(nj)*.84))
         vel1=vs(1)
         sig1=fw(1)/2.35
         h31=h3a(1)
         h41=h4a(1)
         call biwgt(vs,nj,xbv,xs)
         call biwgt(fw,nj,xbf,xs)
         call biwgt(h3a,nj,xbh3,xs)
         call biwgt(h4a,nj,xbh4,xs)
         biasv=vel1-xbv
         biass=sig1-xbf/2.35
         biash3=h31-xbh3
         biash4=h41-xbh4
         rad(nf)=rsim
         vel(nf)=xbv+biasv
         vell(nf)=vs(i16)+biasv
         velh(nf)=vs(i84)+biasv
         sig(nf)=xbf/2.35+biass
         write(*,1002) biass,xbf/2.35,biash3,biash4
c         print *,nj,sig1,sig(nf)
         sigl(nf)=fw(i16)/2.35+biass
         sigh(nf)=fw(i84)/2.35+biass
         h3(nf)=xbh3+biash3
         h3l(nf)=h3a(i16)+biash3
         h3h(nf)=h3a(i84)+biash3
         h4(nf)=xbh4+biash4
         h4l(nf)=h4a(i16)+biash4
         h4h(nf)=h4a(i84)+biash4
         do i=1,ns
            do j=1,nj
               yin(j)=stype1(i,j)
            enddo
            call biwgt(yin,nj,xb,xs)
            sum=0.
            do j=1,nj
               sum=sum+yin(j)
            enddo
            stype(nf,i)=sum/float(nj)
         enddo
         write(11,1101) rad(nf),vel(nf),velh(nf)-vel(nf),sig(nf),
     $        sigh(nf)-sig(nf),h3(nf),h3h(nf)-h3(nf),
     $        h4(nf),h4h(nf)-h4(nf),file1
         write(*,1101) rad(nf),vel(nf),velh(nf)-vel(nf),sig(nf),
     $        sigh(nf)-sig(nf),h3(nf),h3h(nf)-h3(nf),
     $        h4(nf),h4h(nf)-h4(nf),file1
         write(13,1301) rad(nf),nint(vel(nf)),nint(vell(nf)),
     $        nint(velh(nf)),nint(sig(nf)),nint(sigl(nf)),
     $        nint(sigh(nf)),h3(nf),h3l(nf),h3h(nf),
     $        h4(nf),h4l(nf),h4h(nf),file1
         radmin=min(radmin,rad(nf))
         radmax=max(radmax,rad(nf))
         velmin=min(velmin,vell(nf))
         velmax=max(velmax,velh(nf))
         sigmin=min(sigmin,sigl(nf))
         sigmax=max(sigmax,sigh(nf))
 767     continue
      enddo
 666  continue
      close(1)
      close(12)
      
      radbit=(radmax-radmin)/20.
      radmin=radmin-radbit
      radmax=radmax+radbit
      velbit=(velmax-velmin)/20.
      velmin=velmin-velbit
      velmax=velmax+velbit
      sigbit=(sigmax-sigmin)/20.
      sigmin=sigmin-sigbit
      sigmax=sigmax+sigbit

c      velmin=-230
c      velmax=230.
c      radmin=-1.1
c      radmax=1.1

      call pgenv(radmin,radmax,velmin,velmax,0,0)
      call pgpoint(nf,rad,vel,17)
      call pgerry(nf,rad,velh,vell,1.)
      call pglabel('Arcsec','Velocity','')

      call pgenv(radmin,radmax,sigmin,sigmax,0,0)
      call pgpoint(nf,rad,sig,17)
      call pgerry(nf,rad,sigh,sigl,1.)
      call pglabel('Arcsec','Dispersion','')

      close(11)
      close(13)
 1101 format(f7.2,1x,4(f8.3,1x,f7.3),2x,a20)
 1301 format(f7.2,1x,6(i6,1x),6(f8.3,1x),2x,a8)
 2001 format(f7.1,3(1x,f8.3))

c      call pgenv(radmin,radmax,0.,1.0,0,0)
      iskip=6
      nct=0
      do i=1,ns,iskip
         nct=nct+1
         do j=1,nf
            sum=0.
            do k=i,min(ns,i+iskip-1)
               sum=sum+stype(j,k)
            enddo
            yin(j)=sum
         enddo
         if(nct.eq.1) then 
c            call pgpoint(nf,rad,yin,17)
         elseif(nct.eq.2) then
c            call pgpoint(nf,rad,yin,13)
         elseif(nct.eq.2) then
c            call pgpoint(nf,rad,yin,-4)
         elseif(nct.eq.3) then
c            call pgpoint(nf,rad,yin,6)
         elseif(nct.ge.4) then
c            call pgpoint(nf,rad,yin,23)
         endif
      enddo
c      call pglabel('Arcsec','Spectral Type','')

      call pgend

      end

      subroutine getvell(n,x,y,xmax,ymax,xlow,xhigh)
      real x(n),y(n)

      do i=1,n-1
         if(ymax.gt.y(i).and.ymax.le.y(i+1)) then
            xlow=x(i)+(x(i+1)-x(i))*(ymax-y(i))/(y(i+1)-y(i))
         endif
         if(ymax.le.y(i).and.ymax.gt.y(i+1)) then
            xhigh=x(i)+(x(i+1)-x(i))*(ymax-y(i))/(y(i+1)-y(i))
         endif
      enddo
      return
      end

      subroutine getfwhm(n,x,y,frac,fwhm,xmax2,ymax2,v1,v2)
      real x(n),y(n),y2(10000)

      data big /1.e20/

c      call spline(x,y,n,0.,0.,y2)

      ymax=-big
      ymax2=-big
      do i=1,n-1
         do ia=1,9
            xp=x(i)+float(ia-1)/9.*(x(i+1)-x(i))
c            call splint(x,y,y2,n,xp,yp)
            if(yp.gt.ymax2) then
               ymax2=yp
               xmax2=xp
            endif
         enddo
         if(y(i).gt.ymax) then
            ymax=y(i)
            imax=i
         endif
      enddo

      ymax2=ymax
      xmax2=x(imax)
      yhalf=ymax2*frac

      diff=big
      x1=x(1)
      do i=1,imax-1
         if(yhalf.ge.y(i).and.yhalf.lt.y(i+1)) then
            x1=x(i)+(yhalf-y(i))/(y(i+1)-y(i))*(x(i+1)-x(i))
         endif
      enddo

      diff=big
      x2=x(n)
      do i=imax,n-1
         if(yhalf.ge.y(i+1).and.yhalf.lt.y(i)) then
            x2=x(i+1)+(yhalf-y(i+1))/(y(i)-y(i+1))*(x(i)-x(i+1))
         endif
      enddo

      fwhm=x2-x1

c - get 1st and second moment:

      sum1=0.
      sum2=0.
      do i=1,n
         sum1=sum1+x(i)*y(i)
         sum2=sum2+y(i)
      enddo
      v1=sum1/sum2
      sum1=0.
      sum2=0.
      do i=1,n
         sum1=sum1+y(i)*(x(i)-v1)**2
         sum2=sum2+y(i)
      enddo
      v2=sqrt(sum1/sum2)

      return
      end
