
      parameter(nmax=1000)

      integer ida1(nmax),ida2(nmax)
      character file1*80,a1*3,a2*2,ba1(nmax)*3,ba2(nmax)*2
      character c1*80,c2*80,c3*80,camp*3,camp2*2

      open(unit=1,file='/work/03946/hetdex/hdr1/reduction/badamps.list',
     $     status='old')
      n=0
      do i=1,nmax
         read(1,*,end=666) a1,a2,id1,id2
         n=n+1
         ba1(n)=a1
         ba2(n)=a2
         ida1(n)=id1
         ida2(n)=id2
      enddo
 666  continue
      close(1)

      open(unit=11,file='out',status='unknown')
      open(unit=2,file='inbad',status='old')
      do j=1,10000
         read(2,*,end=888) file1
         open(unit=1,file=file1,status='old')
         rmin=1e10
         do i=1,10000
            read(1,*,end=667) x1,x2,x3,x4,c1,c2,x7,x8,c3,i9,i10
            if(x7.lt.rmin) then
               rmin=x7
               camp=c1(11:13)
               camp2=c1(19:20)
               idate=i9
            endif
         enddo
 667     continue
         close(1)

         do i=1,n
            if(idate.ge.ida1(i).and.idate.le.ida2(i)) then
               if(camp.eq.ba1(i).and.camp2.eq.ba2(i)) then
                  write(11,1101) file1(1:30),camp,camp2
                  goto 777
               endif
            endif
         enddo
 777     continue
      enddo
 888  continue
      close(2)
      close(11)

 1101 format(a30,1x,a3,1x,a2)

      end
