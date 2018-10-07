
      parameter(narr=1000,pi=3.14159)
      real x(narr),y(narr),xsum(narr,narr),tr(6),xfib(narr),yfib(narr)
      real xfib2(narr),yfib2(narr)
      real xsumb(narr,narr),xsumr(narr,narr),xsum2(narr,narr)
      integer nsum(narr,narr)

      xfwhm=1.7
      rsig=xfwhm/2.35
      vlo=-30
      vup=200
      xs=-4.
      xe=4.
      xb=0.5
      xb=0.25
      n=nint((xe-xs)/xb)+1
      print *,n
      do i=1,n
         x(i)=xs+float(i-1)*xb
         y(i)=xs+float(i-1)*xb
c         print *,i,x(i),y(i)
      enddo

      do i=1,n
         do j=1,n
            xsum(i,j)=0.
            xsum2(i,j)=0.
            xsumb(i,j)=0.
            xsumr(i,j)=0.
            nsum(i,j)=0
         enddo
      enddo

      open(unit=1,file='in',status='old')
      open(unit=2,file='in2',status='old')

      xtot=0.
      ntot=0
      nfib=0
      do ia=1,10000
         read(1,*,end=666) x1,x2,x3,x4,x5
         radm=1e10
         sum1=0.
         nfib=nfib+1
         xfib(nfib)=x1
         yfib(nfib)=x2
         do i=1,n
            do j=1,n
               xp=(x1-x(i))
               yp=(x2-y(j))
               rad=sqrt(xp*xp+yp*yp)
               if(rad.lt.radm) then
                  radm=rad
                  ix=i
                  iy=j
               endif
               g=rad/rsig
               gaus=exp(-g*g/2.)/(2.*rsig*rsig*pi)
               xsum(i,j)=xsum(i,j)+x4*gaus
               xsumb(i,j)=xsumb(i,j)+x3*gaus
               xsumr(i,j)=xsumr(i,j)+x5*gaus
               sum1=sum1+x4*gaus
            enddo
         enddo
c         print *,ia,x4,sum1
c         xsum(ix,iy)=xsum(ix,iy)+x4
c         xsumb(ix,iy)=xsumb(ix,iy)+x3
c         xsumr(ix,iy)=xsumr(ix,iy)+x5
         nsum(ix,iy)=nsum(ix,iy)+1
         xtot=xtot+x3+x5
         ntot=ntot+2
      enddo
 666  continue
      close(1)
      xtot=xtot/float(ntot)
      print *,xtot

      nfib2=0
      do ia=1,10000
         read(2,*,end=667) x1,x2,x3,x4,x5
         radm=1e10
         nfib2=nfib2+1
         xfib2(nfib2)=x1
         yfib2(nfib2)=x2
         do i=1,n
            do j=1,n
               xp=(x1-x(i))
               yp=(x2-y(j))
               rad=sqrt(xp*xp+yp*yp)
               if(rad.lt.radm) then
                  radm=rad
                  ix=i
                  iy=j
               endif
               g=rad/rsig
               gaus=exp(-g*g/2.)/(2.*rsig*rsig*pi)
               xsum2(i,j)=xsum2(i,j)+x4*gaus
            enddo
         enddo
c         xsum2(ix,iy)=xsum2(ix,iy)+x4
      enddo
 667  continue
      close(2)

      vmin=1e10
      vmax=-1e10
      do i=1,n
         do j=1,n
            xsum(i,j)=xsum(i,j)-xtot
            xsum2(i,j)=xsum2(i,j)-xtot
            xsumb(i,j)=xsumb(i,j)-xtot
            xsumr(i,j)=xsumr(i,j)-xtot
            vmax=max(vmax,xsum(i,j),xsum2(i,j),xsumb(i,j),xsumr(i,j))
            vmin=min(vmin,xsum(i,j),xsum2(i,j),xsumb(i,j),xsumr(i,j))
         enddo
      enddo
      print *,vmin,vmax
      vup=min(vup,vmax)

      open(unit=11,file='out',status='unknown')
      do i=1,n
         do j=1,n
            if(nsum(i,j).gt.0) then
               write(11,*) x(i),y(j),xsum(i,j),nsum(i,j)
            else
c               xsum(i,j)=0
            endif
         enddo
      enddo
      close(11)

      tr(1)=-float(n-1)/2.*xb
      tr(2)=xb
      print *,tr(1),tr(2)
      tr(3)=0
      tr(4)=-float(n-1)/2.*xb
      tr(5)=0
      tr(6)=xb

      call pgbegin(0,'?',1,1)
      call pgpap(0.0,1.0)
      call pgscf(2)
      call pgslw(3)
      call pgsch(1.2)

      call pgvport(0.05,0.45,0.55,0.95)
      call pgwindow(xs,xe,xs,xe)
      call pgbox('bcnst',0.,0,'bcnst',0.,0)
      call pggray(xsum,narr,narr,1,n,1,n,vup,vlo,tr)
      call pgsch(0.9)
      call pgsci(2)
      call pgpt(nfib,xfib,yfib,17)
      call pgsci(1)
      call pgsch(1.2)

      call pgvport(0.5,0.9,0.55,0.95)
      call pgwindow(xs,xe,xs,xe)
      call pgbox('bcnst',0.,0,'bct',0.,0)
      call pggray(xsum2,narr,narr,1,n,1,n,vup,vlo,tr)
      call pgsch(0.9)
      call pgsci(2)
      call pgpt(nfib2,xfib2,yfib2,17)
      call pgsci(1)
      call pgsch(1.2)

      call pgvport(0.05,0.45,0.08,0.48)
      call pgwindow(xs,xe,xs,xe)
      call pgbox('bcnst',0.,0,'bcnst',0.,0)
      call pggray(xsumb,narr,narr,1,n,1,n,vup,vlo,tr)

      call pgvport(0.5,0.9,0.08,0.48)
      call pgwindow(xs,xe,xs,xe)
      call pgbox('bcnst',0.,0,'bct',0.,0)
      call pggray(xsumr,narr,narr,1,n,1,n,vup,vlo,tr)


      end
