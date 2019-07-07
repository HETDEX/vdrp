
      parameter(nmax=1500,nf=1000)
      real w(nf,nmax),f1(nf,nmax),f1e(nf,nmax),f2(nf,nmax),f2e(nf,nmax)
      real f3(nf,nmax),f3e(nf,nmax),w1(nf,nmax),ac(nf,nmax)
      character fname*50

      open(unit=1,file='list',status='old')
      n2=0
      do i=1,nf
         read(1,*,end=666) fname
         open(unit=2,file=fname,status='old')
         n2=n2+1
         n=0
         do j=1,nmax
            read(2,*,end=667) x1,x2,x3,x4,x5,x6,x7,x8,x9
            n=n+1
            w(i,n)=x1
            f1(i,n)=x2
            f1e(i,n)=x3
            f2(i,n)=x4
            f2e(i,n)=x5
            f3(i,n)=x6
            f3e(i,n)=x7
            w1(i,n)=x8
            ac(i,n)=x9
         enddo
 667     continue
         close(2)
      enddo
 666  continue
      close(1)

      open(unit=11,file='spec.out',status='unknown')
      do i=1,n
         s1=0.
         s2=0.
         s3=0.
         s4=0.
         s5=0.
         s6=0.
         s7=0.
         s8=0.
         do j=1,n2
            s1=s1+f1(j,i)
            s2=s2+f1e(j,i)**2
            s3=s3+f2(j,i)
            s4=s4+f2e(j,i)**2
            s5=s5+f3(j,i)
            s6=s6+f3e(j,i)**2
            s7=s7+w1(j,i)
            s8=s8+ac(j,i)
         enddo
         s1=s1/float(n2)
         s2=sqrt(s2)/float(n2)
         s3=s3/float(n2)
         s4=sqrt(s4)/float(n2)
         s5=s5/float(n2)
         s6=sqrt(s6)/float(n2)
         s7=s7/float(n2)
         s8=s8/float(n2)
         write(11,1101) w(1,i),s1,s2,s3,s4,s5,s6,s7,s8
      enddo
 1101 format(1x,f7.2,8(1x,f11.3))

      end
