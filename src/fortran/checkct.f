
      real fb(10),snsum(10),fsum(10)
      integer ncb(10),nc0(10)

      dist0=2.
      wd0=3.
      open(unit=1,file="all.dat",status='old')
      open(unit=2,file="in",status='old')

      nb=10
      fb(1)=000.
      fb(2)=200.
      fb(3)=300.
      fb(4)=400.
      fb(5)=500.
      fb(6)=600.
      fb(7)=700.
      fb(8)=800.
      fb(9)=900.
      fb(10)=10000.
      do i=1,nb
         ncb(i)=0
         nc0(i)=0
         snsum(i)=0.
         fsum(i)=0.
      enddo

      nc=0
      do i=1,1000
         read(2,*,end=666) x,y,w,xf
         do ib=1,nb-1
            if(xf.ge.fb(ib).and.xf.lt.fb(ib+1)) then
               nc0(ib)=nc0(ib)+1
            endif
         enddo                  
         do j=1,10000
            read(1,*,end=667) x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12
            dist=sqrt((x11-x)**2+(x12-y)**2)
            wd=abs(x3-w)
            if(dist.le.dist0.and.wd.le.wd0) then
               nc=nc+1
               print *,nc,x3,w,xf,x4
               do ib=1,nb-1
                  if(xf.ge.fb(ib).and.xf.lt.fb(ib+1)) then
                     ncb(ib)=ncb(ib)+1
                     snsum(ib)=snsum(ib)+x4
                     fsum(ib)=fsum(ib)+x6
                  endif
               enddo                  
            endif
         enddo
 667     continue
         rewind(1)         
      enddo
 666  continue
      close(2)
      close(1)
      
      do i=1,nb-1
         if(nc0(i).gt.0) then
            rat=float(ncb(i))/float(nc0(i))
         else
            rat=0.
         endif
         if(ncb(i).gt.0) then
            rat2=snsum(i)/float(ncb(i))
            rat3=fsum(i)/float(ncb(i))
         else
            rat2=0.
            rat3=0.
         endif
         write(*,1001) i,rat,ncb(i),nc0(i),fb(i)+50.,rat2,rat3
      enddo
 1001 format(i2,1x,f6.2,1x,i4,1x,i4,1x,f6.1,1x,f6.2,1x,f7.2)
      end
