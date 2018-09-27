      subroutine fcn(n,a,func)
      parameter(nmax=50200,nlovm=600,nfile=20)
      real a(n),gp(nmax),t(nmax,nfile),glog(nmax),x(nmax),gerr(nmax)
      real fac(nlovm),temp(nmax),b(nlovm),b2(nlovm),xlama(nlovm)
      real facnew(nlovm*4),ynew(nlovm*4),xl(nlovm),xsum(nmax)
      common /cfunc/ t,glog,gerr,x
      common /cfunc2/ fac,c1,resd,xlam,xl,sigl0,amp1,
     $     np,nl,iskip,nt,ntot,ilog,isym,icoff,nlosvd,igh,igaus,
     $     coffi,coff2i

      parameter(cee=2.99e+5,pi=3.141593)

      ntot=ntot+1

c - take care of symmetry

      ibnt=nl
      if(igh.eq.1) then
         ibnt=5
         if(igaus.eq.1) ibnt=3
      endif
      if(isym.eq.1) then
         ibnt=(nl+1)/2
         imid=ibnt
         do i=1,nl
            if(i.lt.imid) then
               b(i)=a(imid+1-i)
            else
               b(i)=a(i-imid+1)
            endif
            b(i)=max(b(i),1.e-6)
         enddo
      else
         if(igh.eq.0) then
            do i=1,nl
               b(i)=a(i)
               b(i)=max(b(i),1.e-6)
            enddo
         else
            amp=a(1)*amp1
            vel=a(2)*sigl0
            sig=a(3)*sigl0
            if(igaus.eq.0) then
               h3=a(4)
               h4=a(5)
            else
               h3=0.
               h4=0.
            endif
            den=sqrt(2.*pi)
            do i=1,nl
               w=(xl(i)-vel)/sig
               gaus=exp(-w*w/2.)/den
               b(i)=amp*gaus/sig*(1.+h3*fh3(w)+h4*fh4(w))
            enddo
         endif
      endif

      if(icoff.eq.0) then
         coff=coffi
         coff2=coff2i
      endif
      if(icoff.eq.1) then
         coff=a(ibnt+nt+1)
         coff2=a(ibnt+nt+2)
      endif
      if(icoff.eq.2) then
         coff=coffi
         coff2=a(ibnt+nt+1)
      endif

c - shift the symmetric profile

c      if(isym.eq.1) then
c         xshift=a(ibnt+nt+1)*500.
c         do i=1,nl
c            call xlinint(xl(i)-xshift,nl,xl,b,b2(i))
c         enddo
c         do i=1,nl
c            b(i)=b2(i)
c         enddo
c      endif

c - get the sum for normalization

      sumb=0.
      do i=1,nl
         sumb=sumb+b(i)
      enddo

c - do the convolution

      sum2=0.
      do i=1,nt
         sum2=sum2+a(ibnt+i)
      enddo

      do i=1,np
         gp(i)=0.
         xsum(i)=0.
         sum1=0.
         sum2o=sum2
         do it=1,nt
            if(t(i,it).ne.1) then
               sum1=sum1+a(ibnt+it)*t(i,it)
            else
               sum2o=sum2o-a(ibnt+it)
            endif
         enddo
         if(sum2o.eq.0) sum2o=1.
         tval=sum1/sum2o
         temp(i)=(tval+coff)/(coff+1.)+coff2
      enddo

      call getnlosvd(nl,xl,b,nlosvd,facnew,ynew)

      do i=1,np
        do j=1,nlosvd
           ip=nint(c1+x(i)*facnew(j))
           if(ip.gt.0.and.ip.le.np) then
              gp(ip)=gp(ip)+temp(i)*ynew(j)
              xsum(ip)=xsum(ip)+ynew(j)
           endif
        enddo
      enddo
      suml=0.
      do i=1,nlosvd
         suml=suml+ynew(i)
      enddo
      do i=1,np
         gp(i)=gp(i)*suml/xsum(i)
      enddo
c      do i=1,np
c         gp(i)=gp(i)/xsum(i)
c      enddo

c - get the contribution from the difference
      
      sum1=0.
      if(ilog.eq.1) then
         do i=iskip+1,np-iskip
            sum1=sum1+((glog(i)-log(gp(i)))/gerr(i))**2
         enddo
      else
         do i=iskip+1,np-iskip
            sum1=sum1+((glog(i)-(gp(i)))/gerr(i))**2
         enddo
      endif      

c - get the contribution from the smoothness

      sfac=1.8
      do i=1,nl
         xlama(i)=xlam*1.0
c         if(i.gt.9) xlama(i)=xlam*300.0
         if(abs(xl(i)).gt.sfac*sigl0) then
            xlama(i)=xlam*(abs(xl(i)/sigl0)/sfac)**4
         endif
c         print *,i,xl(i),xlama(i)
      enddo

      sum2=0.
      if(igh.eq.0) then

c - asymptotic straight line smoothing

         do i=2,nl-1
            sval=(b(i+1)-2.*b(i)+b(i-1))**2
            sum2=sum2+xlama(i)*sval
c            print *,i,xlama(i),sval
         enddo
         sval1=xlama(1)*(b(2)-2.*b(1))**2
         sval2=xlama(nl)*(b(nl-1)-2.*b(nl))**2
c         print *,1,xlama(1),(b(2)-2.*b(1))**2
c         print *,nl,xlama(nl),(b(nl-1)-2.*b(nl))**2
         sum2=sum2+sval1+sval2
         sum2=sum2/resd

      endif

      fadd=1.e-1*abs(sumb-1.)
      func=sum1+sum2+fadd

      return
      end

      subroutine fcnsym(n,a,func)
      parameter(nmax=50200,nlovm=600,nfile=20)
      real a(n),gp(nmax),t(nmax,nfile),glog(nmax),x(nmax),gerr(nmax)
      real fac(nlovm),temp(nmax),b(nlovm),gp2(nmax),ynew2(nlovm*4)
      real facnew(nlovm*4),ynew(nlovm*4),xl(nlovm),b2(nlovm)
      real glog2(nmax),gerr2(nmax),temp2(nmax),xsum(nmax)

      common /cfuncsym/ t,glog,gerr,glog2,gerr2,
     $     x,fac,c1,resd,xlam,xl,sigl0,amp1,
     $     np,nl,iskip,nt,ntot,ilog,isym,icoff,nlosvd,igh,igaus,
     $     coffi,coff2i

      parameter(cee=2.99e+5,pi=3.141593)

      ntot=ntot+1

c - take care of symmetry

      ibnt=nl
      if(igh.eq.1) then
         ibnt=5
         if(igaus.eq.1) ibnt=3
      endif
      if(isym.eq.1) then
         ibnt=(nl+1)/2
         imid=ibnt
         do i=1,nl
            if(i.lt.imid) then
               b(i)=a(imid+1-i)
            else
               b(i)=a(i-imid+1)
            endif
         enddo
      else
         if(igh.eq.0) then
            do i=1,nl
               b(i)=a(i)
               b(i)=max(b(i),1.e-6)
            enddo
         else
            amp=a(1)*amp1
            vel=a(2)*sigl0
            sig=a(3)*sigl0
            if(igaus.eq.0) then
               h3=a(4)
               h4=a(5)
            else
               h3=0.
               h4=0.
            endif
            den=sqrt(2.*pi)
            do i=1,nl
               w=(xl(i)-vel)/sig
               gaus=exp(-w*w/2.)/den
               b(i)=amp*gaus/sig*(1.+h3*fh3(w)+h4*fh4(w))
            enddo
         endif
      endif

      if(icoff.eq.0) then
         coff=coffi
         coff2=coff2i
      endif
      if(icoff.eq.1.and.isym.eq.0) then
         coff=a(ibnt+nt+1)
         coff2=a(ibnt+nt+2)
      endif
      if(icoff.eq.1.and.isym.eq.1) then
         coff=a(ibnt+nt+2)
         coff2=a(ibnt+nt+3)
      endif
      if(icoff.eq.2.and.isym.eq.0) then
         coff=coffi
         coff2=a(ibnt+nt+1)
      endif
      if(icoff.eq.2.and.isym.eq.1) then
         coff=coffi
         coff2=a(ibnt+nt+2)
      endif

c - shift the symmetric profile

      if(isym.eq.1) then
         xshift=a(ibnt+nt+1)*500.
         do i=1,nl
            call xlinint(xl(i)-xshift,nl,xl,b,b2(i))
         enddo
         do i=1,nl
            b(i)=b2(i)
         enddo
      endif

c - do the convolution

      sum2=0.
      do i=1,nt
         sum2=sum2+a(ibnt+i)
      enddo

      do i=1,np
         gp(i)=0.
         gp2(i)=0.
         sum1=0.
         sum2o=sum2
         do it=1,nt
            if(t(i,it).ne.1) then
               sum1=sum1+a(ibnt+it)*t(i,it)
            else
               sum2o=sum2o-a(ibnt+it)
            endif
         enddo
         if(sum2o.eq.0) sum2o=1.
         tval=sum1/sum2o
         temp(i)=(tval+coff)/(coff+1.)+coff2
      enddo

      call getnlosvd(nl,xl,b,nlosvd,facnew,ynew)
      call invert(nlosvd,ynew,ynew2)

      do i=1,np
         do j=1,nlosvd
           ip=nint(c1+x(i)*facnew(j))
           if(ip.gt.0.and.ip.le.np) then
              gp(ip)=gp(ip)+temp(i)*ynew(j)
              gp2(ip)=gp2(ip)+temp(i)*ynew2(j)
              xsum(ip)=xsum(ip)+ynew(j)
           endif
        enddo
      enddo

c - get the contribution from the difference
      
      sum1=0.
      if(ilog.eq.1) then
         do i=iskip+1,np-iskip
            gpp1=glog(i)
            gpp1=glog2(i)
            sum1=sum1+((gpp1-log(gp(i)))/gerr(i))**2+
     $                ((gpp2-log(gp2(i)))/gerr2(i))**2
         enddo
      else
         do i=iskip+1,np-iskip
            gpp1=glog(i)
            gpp1=glog2(i)
            sum1=sum1+((glog(i)-(gp(i)))/gerr(i))**2+
     $                ((glog2(i)-(gp2(i)))/gerr2(i))**2
         enddo
      endif      

c - get the contribution from the smoothness

      sum2=0.
      if(igh.eq.0) then

c - asymptotic straight line smoothing

         do i=2,nl-1
            sum2=sum2+(b(i+1)-2.*b(i)+b(i-1))**2
         enddo
         sum2=sum2+(-2.*b(nl)+b(nl-1))**2+(b(2)-2.*b(1))**2
         sum2=sum2/resd
      endif

      func=sum1+xlam*sum2
      if(igh.eq.0) then
c         write(*,"(1x,i6,3(1x,f9.5),1x,f6.3,1x,f8.3,a1,$)") ntot,sum1,
c     $        xlam*sum2,func,coff,xshift,char(13)
      else
c         write(*,"(1x,i6,7(1x,f9.4),a1,$)") ntot,sum1,
c     $        amp,vel,sig,h3,h4,coff,char(13)
      endif
c      call flush(6)

      return
      end

      subroutine getnlosvd(n,x,y,n2,x2,y2)
      real x(n),y(n),x2(n2),y2(n2)
      common /cscale/ scale
      parameter(cee=2.99e+5)

      xmin=x(1)
      xmax=x(n)
      xdiff=(xmax-xmin)/float(n2-1)
      sumo=0.
      do i=1,n
         sumo=sumo+y(i)
      enddo
      sum=0.
      do i=1,n2
         x2p=xmin+xdiff*(i-1)
         if(x2p.le.xmin) then
            y2(i)=y(1)
         elseif(x2p.ge.xmax) then
            y2(i)=y(n)
         else
            do j=1,n-1
               if(x2p.gt.x(j).and.x2p.le.x(j+1)) then
                  y2(i)=y(j)+(y(j+1)-y(j))*(x2p-x(j))/(x(j+1)-x(j))
               endif
            enddo
         endif
         x2(i)=(1.+x2p/cee)/scale
         sum=sum+y2(i)
      enddo
      frac=sumo/sum
c      frac=1.
      do i=1,n2
         y2(i)=y2(i)*frac
      enddo

      return
      end

      subroutine setbadreg(n,x,xe,iflip)
      real x(n),xe(n)
      data big/1.e10/
      open(unit=11,file='regions.bad',status='old',err=766)
      do i=1,1000
         read(11,*,end=666) x1,x2
         do j=1,n
            if(x(j).ge.iflip*x1.and.x(j).le.iflip*x2) xe(j)=big
         enddo
      enddo
 666  continue
      close(11)
      return
 766  print *
      print *,'No regions.bad file!'
      print *
      return
      end

      subroutine geti(file1,ncol,nrow,x)
      parameter(narr=50200)
      real x(narr,1)
      integer naxes(2)
      character file1*80
      logical simple,extend,anyf
      ier=0
      im1=50
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         print *,ier
         write(*,*) 'Error opening image : ',file1
         ier=0
         goto 706
      endif
c      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxis.eq.1) naxes(2)=1
      if(naxes(1).gt.narr.or.naxes(2).gt.narr) then
         write(*,"('Arrays too small - make narr bigger')")
         write(*,"('Axes equal to ')") naxes(1),naxes(2)
         goto 706
      endif
      ncol=naxes(1)
      nrow=max(1,naxes(2))
      call ftg2de(im1,igc,0.,narr,ncol,nrow,x,anyf,ier)
      call ftclos(im1,ier)
      return
 706  print *,'ier = ',ier
c      pause 'Die'
      return
      end

      subroutine plotline(n,x,y,ye)
      parameter(nmax=50200)
      real x(n),y(n),ye(n),xp(nmax),yp(nmax)
      data big/1.e10/
      ystart=ye(1)
      istart=1
      idone=0
      do j=1,nmax
         ibad=1
         if(ystart.lt.big) ibad=0
         np=0
         do i=istart,n
            if(ye(i).eq.ystart) then
               np=np+1
               xp(np)=x(i)
               yp(np)=y(i)
            else
               ystart=ye(i)
               istart=i
               goto 666
            endif
         enddo
         idone=1
 666     if(ibad.eq.1) then
            call pgsls(4)
            call pgline(np,xp,yp)
            call pgsls(1)
         else
            call pgline(np,xp,yp)
         endif
         if(idone.eq.1) goto 766
      enddo
 766  continue
 
      return
      end

      subroutine xlinint(xp,n,x,y,yp)
      real x(n),y(n)
      do j=1,n-1
         if(xp.ge.x(j).and.xp.lt.x(j+1)) then
            yp=y(j)+(y(j+1)-y(j))*(xp-x(j))/(x(j+1)-x(j))
            return
         endif
      enddo
      if(xp.lt.x(1)) yp=y(1)
      if(xp.gt.x(n)) yp=y(n)
      return
      end

      subroutine setinitv(sigl,ibnt,yl0,xscale,xlb,xub)
      real yl0(ibnt),xscale(ibnt),xlb(ibnt),xub(ibnt)
      yl0(1)=1.
      yl0(2)=0.
      yl0(3)=sigl
      yl0(4)=0.
      yl0(5)=0.
      xscale(1)=1.
      xscale(2)=0.1
      xscale(3)=0.02
      xscale(4)=1.
      xscale(5)=1.
      xlb(1)=0.7
      xub(1)=1.5
      xlb(2)=-2.5*sigl
      xub(2)=+2.5*sigl
c      xlb(2)=-1.
c      xub(2)=+1
      xlb(3)=sigl/3.
      xub(3)=sigl*3.
c      xlb(3)=sigl/1.01
c      xub(3)=sigl*1.01
      xlb(4)=-0.3
      xub(4)=+0.3
      xlb(5)=-0.3
      xub(5)=+0.3
      return
      end

      function fh3(x)
      fh3=1./sqrt(6.)*(2.*sqrt(2.)*x*x*x-3.*sqrt(2.)*x)
      return
      end
      function fh4(x)
      fh4=1./sqrt(24.)*(4.*x*x*x*x-12.*x*x+3.)
      return
      end

      subroutine invert(n,x,x2)
      real x(n),x2(n)
      do i=1,n
         x2(i)=x(n-i+1)
      enddo
      return
      end
