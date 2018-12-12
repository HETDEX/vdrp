
      parameter(narrm=6000,nmax=100)
      parameter(pi=3.141592e0)
      real xd(narrm,narrm),xr(narrm,narrm)
      real xr3(narrm,narrm),xdo(narrm,narrm)
      real flux(nmax),rad(nmax),flux2(nmax),eflux(nmax)

      integer naxes(2)
      character file1*40
      logical simple,extend,anyf
      external func

      common /data/ xd,xr,xr3,xdo
      common /ciget/ smin,smax,smin2,ntot
      common /ab/ a,b
      common /cdisk/ d0,rd,cosi2,xang,dexp

      data big/1.e20/

      call qr2('Min and Max radii ','iso.def',smin2,smax)
      smin=smin2
      call qi1('Number of bins ','iso.def',ntot)
      call qi1('Number of steps ','iso.def',ns)
      call qr3('Center coord and step ','iso.def',xc,yc,pstep)
      call qr2('Axis ratio and step ','iso.def',axr,astep)
      call qr2('PA and step ','iso.def',pa,pastep)
      call qi1('Fit disk (1-yes) ','iso.def',idisk)
      d0=0
      rd=1.
      xinc=1.
      d0step=0.
      rdstep=0.
      xistep=0.
      destep=0.
      if(idisk.eq.1) then
         call qr2('I_0 and step for disk ','iso.def',d0,d0step)
         call qr2('R_d and step for disk ','iso.def',rd,rdstep)
         call qr2('Inc and step for disk ','iso.def',xi,xistep)
         call qr2('Exp and step for disk ','iso.def',de,destep)
         call qr1('Angle up from mj to ignore ','iso.def',xang)
         xang=xang*pi/180.
      endif
 1    call qc1('Image ','iso.def',file1)
      call savdef

      a=rtbiso(func,.00001,0.9,1.e-6)
      b=a*smin/(exp(a)-1.0)
      write(*,*) a,b
      write(*,*) 'smax = ',b/a*(exp(a*ntot)-1.0)

      ier=0
      im1=0
      call ftgiou(im1,ier)
      iread=0
      iblock=1
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file1
         goto 1
      endif
c      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxes(1).gt.narrm.or.naxes(2).gt.narrm) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

      do i=1,ncol
         do j=1,nrow
            xr(i,j)=0.
            xdo(i,j)=xd(i,j)
         enddo
      enddo

      open(unit=1,file='pix.bad',status='old',err=866)
      do i=1,100000
         read(1,*,end=866) i1,i2
         xd(i1,i2)=-666
      enddo
 866  continue
      close(1)

c      ns=3
      bit=1.e-4
      xcmin=xc-float(ns-1)/2.*pstep
      xcmax=xcmin+pstep*(ns-1)+bit
      ycmin=yc-float(ns-1)/2.*pstep
      ycmax=ycmin+pstep*(ns-1)+bit
      axrmin=axr-float(ns-1)/2.*astep
      axrmax=axrmin+astep*(ns-1)+bit
      pamin=pa-float(ns-1)/2.*pastep
      pamax=pamin+pastep*(ns-1)+bit
      d0min=d0-float(ns-1)/2.*d0step
      d0max=d0min+d0step*(ns-1)+bit
      rdmin=rd-float(ns-1)/2.*rdstep
      rdmax=rdmin+rdstep*(ns-1)+bit
      ximin=xi-float(ns-1)/2.*xistep
      ximax=ximin+xistep*(ns-1)+bit
      demin=de-float(ns-1)/2.*destep
      demax=demin+destep*(ns-1)+bit

      if(pstep.eq.0) pstep=1
      if(astep.eq.0) astep=1
      if(pastep.eq.0) pastep=1
      if(d0step.eq.0) d0step=1
      if(rdstep.eq.0) rdstep=1
      if(xistep.eq.0) xistep=1
      if(destep.eq.0) destep=1
      summin=big
      open(unit=1,file='iso.iter',status='unknown')
      write(1,*) '       X      Y     ratio  PA     I_D     R_D   ',
     $     'inc_D   n_D       Value'
      write(*,*) '       X      Y     ratio  PA     I_D     R_D   ',
     $     'inc_D   n_D       Value'
      do xc=xcmin,xcmax,pstep
         do yc=ycmin,ycmax,pstep
            do axr=axrmin,axrmax,astep
               do pa=pamin,pamax,pastep
               do xi=ximin,ximax,xistep
               do dexp=demin,demax,destep
               do d0=d0min,d0max,d0step
               do rd=rdmin,rdmax,rdstep
                  cosi2=(cos(xi*pi/180.))**2
                  call getsum(ncol,nrow,xc,yc,axr,pa,sum,rad,flux,flux2,
     $                 eflux)
                  if(sum.lt.summin) then
                     summin=sum
                     xcp=xc
                     ycp=yc
                     axrp=axr
                     pap=pa
                     d0p=d0
                     rdp=rd
                     xip=xi
                     dep=dexp
                     sump=sum
                     write(*,1002) xc,yc,axr,pa,d0,rd,xi,dexp,sum
                     write(1,1002) xc,yc,axr,pa,d0,rd,xi,dexp,sum
                  else
                     write(*,1001) xc,yc,axr,pa,d0,rd,xi,dexp,sum
                     write(1,1001) xc,yc,axr,pa,d0,rd,xi,dexp,sum
                  endif
               enddo
               enddo
               enddo
               enddo
               enddo
            enddo
         enddo
      enddo
 1001 format('    ',2(1x,f7.2),1x,f5.3,1x,f5.1,1x,f7.1,3(1x,f6.2),
     $     1x,f12.3)
 1002 format('BEST',2(1x,f7.2),1x,f5.3,1x,f5.1,1x,f7.1,3(1x,f6.2),
     $     1x,f12.3)
 1003 format('END ',2(1x,f7.2),1x,f5.3,1x,f5.1,1x,f7.1,3(1x,f6.2),
     $     1x,f12.3)

      write(*,1003) xcp,ycp,axrp,pap,d0p,rdp,xip,dep,sump
      write(1,1003) xcp,ycp,axrp,pap,d0p,rdp,xip,dep,sump
      close(1)

      d0=d0p
      rd=rdp
      cosi2=(cos(xip*pi/180.))**2
      dexp=dep
      call getsum(ncol,nrow,xcp,ycp,axrp,pap,sum,rad,flux,flux2,eflux)

      open(unit=1,file='iso.out',status='unknown')
      sum1=0.
      sum2=0.
      do i=1,ntot
         write(1,*) rad(i),flux(i),flux2(i),eflux(i)
         if (i+1.lt.ntot) then
            val=4*pi*rad(i)*rad(i)*(rad(i+1)-rad(i))
            sum1=sum1+val*flux(i)
            sum2=sum2+val*flux2(i)
c            print *,rad(i),sum2/sum1
         endif
      enddo
      close(1)
      print *,'D/Total = ',sum2/sum1

      naxis=2
      im1=0
      ier=0
      call ftinit(50,'iso.fits',iblock,ier)
      call ftphps(50,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(50,igc,narrm,naxes(1),naxes(2),xr,ier)
      call ftclos(50,ier)

      im1=0
      ier=0
      call ftinit(50,'iso3.fits',iblock,ier)
      call ftphps(50,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(50,igc,narrm,naxes(1),naxes(2),xr3,ier)
      call ftclos(50,ier)

 706  continue

      end

      subroutine getsum(ncol,nrow,xc,yc,axr,pa,sum,rad,flux,flux2,eflux)
      parameter(narrm=6000,nmax=100,nt=200*narrm)
      parameter(pi=3.141592e0)
      real xd(narrm,narrm),xr(narrm,narrm)
      real xr3(narrm,narrm),xdo(narrm,narrm)
      real flux(nmax),rad(nmax),flux2(nmax),eflux(nmax)
      real t(nt,nmax),ta(nt,nmax),xin(nt),t2(nt,nmax)
      integer np(nmax),nta(nmax)

      common /data/ xd,xr,xr3,xdo
      common /ciget/ smin,smax,smin2,ntot
      common /ab/ a,b
      common /cdisk/ d0,rd,cosi2,xang,dexp

      do i=1,ntot
         np(i)=0
         nta(i)=0
      enddo

      cpa=cos(pa*pi/180.)
      spa=sin(pa*pi/180.)
      axr2=axr*axr
      do j=1,nrow
         y=float(j)-yc
         do i=1,ncol
            x=float(i)-xc
            xp=x*cpa+y*spa
            yp=-x*spa+y*cpa
            ang=atan(abs(yp/xp))
            s=sqrt(xp*xp+yp*yp/axr2)
            if(s.gt.smin2.and.s.le.smax.and.ang.gt.xang) then
            if(xd(i,j).gt.-400.) then
c            if(xd(i,j).gt.-400..and.yp.lt.-14.) then
               is=iget(s)
               np(is)=np(is)+1
               sd=sqrt(xp*xp+yp*yp/cosi2)
               fluxd=dfunc(d0,sd,rd,dexp)
               res=xd(i,j)-fluxd
               t(np(is),is)=res
               t2(np(is),is)=fluxd
            endif
            endif
         enddo
      enddo

      do i=1,ntot
         call biwgt(t(1,i),np(i),xb,xs)
         flux(i)=xb
c         print *,i,np(i),flux(i),xs/sqrt(float(np(i)))/flux(i)*100.
         eflux(i)=xs/sqrt(float(np(i)))/flux(i)*100.
         call biwgt(t2(1,i),np(i),xb,xs)
         flux2(i)=xb
         rad(i)=b/a*(exp(a*float(i))-1.)
      enddo

      sum=0.
      do j=1,nrow
         y=float(j)-yc
         do i=1,ncol
            xr3(i,j)=0.
            x=float(i)-xc
            xp=x*cpa+y*spa
            yp=-x*spa+y*cpa
            ang=atan(abs(yp/xp))
c            s=max(smin2,sqrt(xp*xp+yp*yp/axr2))
c            if(s.le.smax) then
            s=sqrt(xp*xp+yp*yp/axr2)
            if(s.gt.smin2.and.s.le.smax) then
               call xlinint(s,ntot,rad,flux,fluxv)
               sd=sqrt(xp*xp+yp*yp/cosi2)
               fluxd=dfunc(d0,sd,rd,dexp)
               fluxv=fluxv+fluxd
               is=iget(s)
c               if(xd(i,j).gt.-400..and.ang.gt.xang.and.yp.lt.-14.) then
c               if(xd(i,j).gt.-400..and.fluxv.gt.0.and.ang.gt.xang) then
c                  diff=(log10(xd(i,j))-log10(fluxv))**2
c                  diff=(xd(i,j)-fluxv)**2
                  diff=abs(xd(i,j)-fluxv)
                  sum=sum+diff
                  nta(is)=nta(is)+1
                  ta(nta(is),is)=diff
c               endif
c               xr(i,j)=xd(i,j)-fluxv
               xr(i,j)=xdo(i,j)-fluxv
               xr3(i,j)=fluxv
            else
               xr(i,j)=xd(i,j)
            endif
         enddo
      enddo
      
      sum2=0.
      do i=1,ntot
         do j=1,nta(i)
            xin(j)=ta(j,i)
         enddo
         call biwgt(xin,nta(i),xb,xs)
         if(nta(i).gt.0) sum2=sum2+xb
      enddo
      sum=sum2
      sum=sum/axr

      return
      end

      function iget(x)
      common /ab/ a,b

c- make it evenly spaced in log

c      iget=1+nint((float(ntot)-1.)*(log10(x)-log10(smin))/
c     $     (log10(smax)-log10(smin)))
      iget=int(log(a*x/b+1.)/a+0.5)

      return
      end
      
      function dfunc(d0,sd,rd,dexp)
      dfunc=d0*exp(-(sd/rd)**dexp)
c      print *,-(sd/rd)**dexp
c      print *,dfunc,d0,sd,rd,dexp
      return
      end

      subroutine xlinint(xp,n,x,y,yp)
      real x(n),y(n)
      do j=1,n-1
         if(xp.ge.x(j).and.xp.lt.x(j+1)) then
            yp=y(j)+(y(j+1)-y(j))*(xp-x(j))/(x(j+1)-x(j))
         endif
      enddo
      if(xp.lt.x(1)) yp=y(1)
      if(xp.gt.x(n)) yp=y(n)
      return
      end

      function func(x)
      common /ciget/ smin,smax,smin2,ntot
      func=smin*exp(x*ntot)-smin-smax*exp(x)+smax
c      func=smin*exp(x*ntot)-smin-exp(x)+1.
      return
      end

      FUNCTION rtbiso(func,x1,x2,xacc)
      INTEGER JMAX
      REAL rtbiso,x1,x2,xacc,func
      EXTERNAL func
      PARAMETER (JMAX=40)
      INTEGER j
      REAL dx,f,fmid,xmid
      fmid=func(x2)
      f=func(x1)
      if(f*fmid.ge.0.) pause 'root must be bracketed in rtbis'
      if(f.lt.0.)then
        rtbiso=x1
        dx=x2-x1
      else
        rtbiso=x2
        dx=x1-x2
      endif
      do 11 j=1,JMAX
        dx=dx*.5
        xmid=rtbiso+dx
        fmid=func(xmid)
        if(fmid.le.0.)rtbiso=xmid
        if(abs(dx).lt.xacc .or. fmid.eq.0.) return
11    continue
      pause 'too many bisections in rtbis'
      END
