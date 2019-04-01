
      parameter(nmax=200000)
      real xl(2),yl(2)
      character c1*40,c2*80,c3*80,a1*40,fname*12

      read *,fname
      c1="cat.0"
c      c2="/work/00115/gebhardt/maverick/detect/dexall/"//fname//".cat1"
c      c3="/work/00115/gebhardt/maverick/detect/dexall/"
c     $     //fname//".alines2"

c      c2="/work/00115/gebhardt/maverick/detect/dexall/"//fname//".cat1"
c      c3="/work/05178/cxliu/maverick/detect/dexall/sci/"
c     $     //fname//".alines2"

      c2="/work/00115/gebhardt/maverick/detect/hdr1/"//fname//"/cat.0"
      
      xmin=5.
      xmax=27.
      ymin=0.
      ymax=3.

      sn1=5.3
      sn2=20.
      chi1=1.1
      chi2=3.8

      xl(1)=xmin
      xl(2)=xmax
      yl(1)=chi1+(chi2-chi1)/(sn2-sn1)*(xl(1)-sn1)
      yl(2)=chi1+(chi2-chi1)/(sn2-sn1)*(xl(2)-sn1)
      
      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.5)
      call pgslw(2)

      call pgenv(xmin,xmax,ymin,ymax,0,0)
      call pglabel("S/N","\Gx\U2\D\Dreduced","")

      call pgsch(0.8)
      call pgsci(4)
      open(unit=1,file=c3,status='old',err=668)
      do i=1,nmax
         read(1,*,end=668) x1,x2,x3,x4,x5,x6,x7,x8
         call pgpt1(x4,x5,17)
      enddo
 668  continue
      close(1)

      call pgsch(0.9)
      call pgsci(2)
      open(unit=1,file=c1,status='old',err=666)
      do i=1,nmax
         read(1,*,end=666) a1,x2,x3,x4,x5,x6,x7,x8,x9
         call pgpt1(x5,x6,17)
      enddo
 666  continue
      close(1)

      call pgsch(1.1)
      call pgsci(1)
      open(unit=1,file=c2,status='old',err=667)
      do i=1,nmax
         read(1,*,end=667) a1,x2,x3,x4,x5,x6,x7,x8,x9
         call pgpt1(x5,x6,17)
      enddo
 667  continue
      close(1)

      call pgsci(1)
c      call pgline(2,xl,yl)

      call pgend
      end
