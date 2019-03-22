
      parameter(nmax=10000)
      real x(nmax),y(nmax),xn(nmax),yn(nmax)
      real xb(nmax),yb(nmax),yt(nmax)
      character file1*80,file2*80,c1*18

      ybl0=0.83
      ybu0=0.98
      ystep=0.15

      call pgbegin(0,'?',1,1)
      call pgsch(1.0)
      call pgscf(2)
      call pgslw(3)

      xmin=5150.
c      xmax=5209.
      xmax=5190.
      ymin=0.8
      ymax=1.15

      open(unit=1,file='splist',status='old')

      nl=0
      ic=0
      do il=1,20000
         read(1,*,end=666) file1,c1
c         read(1,*,end=666) file1,xnorm
         open(unit=2,file=file1,status='old',err=888)
         goto 889
 888     continue
         print *,"Does not exist: ",file1
         goto 866
 889     continue
         n=0
         do i=1,nmax
            read(2,*,end=667) x1,x2,x3
            if(x2.ne.0) then
               n=n+1
               x(n)=x1
               y(n)=x2
               yt(n)=x3
            endif
         enddo
 667     continue
         close(2)
         ybl=ybl0-float(il-1)*ystep
         ybu=ybu0-float(il-1)*ystep
         call pgsci(1)
         call pgvport(0.15,0.85,ybl,ybu)
         call pgwindow(xmin,xmax,ymin,ymax)
         call pgsch(1.0)
         call pgbox('bcst',0.,0,'bcnst',0.,0)
         call pgline(n,x,y)
         call pgsci(2)
         call pgline(n,x,yt)
         call pgsci(1)
         call pgptxt(5152.,0.85,0.,0.,c1)
 866     continue
         close(2)
      enddo
 666  continue
      close(1)

      call pgsci(1)
      call pgbox('n',0.,0,'',0.,0)
      call pgsch(1.8)
      call pgmtxt('B',1.5,0.5,0.5,'Wavelength \(2078)')
      call pgmtxt('L',1.15,3.0,0.5,'Relative Flux')

      call pgend

      end
