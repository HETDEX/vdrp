
      parameter(nmax=10000)
      real w(nmax),fin(8,nmax),fout(8,nmax)

      open(unit=1,file='specin',status='old',err=888)
      n=0
      do i=1,nmax
         read(1,*,end=666) x1,x2,x3,x4,x5,x6,x7,x8,x9
         n=n+1
         w(n)=x1
         fin(1,n)=x2
         fin(2,n)=x3
         fin(3,n)=x4
         fin(4,n)=x5
         fin(5,n)=x6
         fin(6,n)=x7
         fin(7,n)=x8
         fin(8,n)=x9
      enddo
 666  continue
      close(1)

      call sclean(n,w,fin,fout)

      open(unit=11,file='out',status='unknown')
      do i=1,n
         write(11,1101) w(i),fout(1,i),fout(2,i),fout(3,i),fout(4,i),
     $        fout(5,i),fout(6,i),fout(7,i),fout(8,i)
      enddo
      close(11)

 888  continue

 1101  format(1x,f8.2,1x,f11.2,1x,1pe11.2,3(1x,0pf7.3),3(1x,1pe11.2))
       end

      subroutine sclean(n,w,f,fout)
      parameter(nmax=10000)
      real w(nmax),f(8,nmax),fout(8,nmax)

      jnear=1
      do i=1,n
         diff=1e10
         do j=1,n
            if(f(1,j).ne.0) then
               d=abs(w(i)-w(j))
               if(d.lt.diff) then
                  diff=d
                  jnear=j
               endif
            endif
         enddo
         fout(1,i)=f(1,jnear)
         fout(2,i)=f(2,jnear)
         fout(3,i)=f(3,i)
         fout(4,i)=f(4,i)
         fout(5,i)=f(5,i)
         fout(6,i)=f(6,jnear)
         fout(7,i)=f(7,jnear)
         fout(8,i)=f(8,jnear)
         if(f(1,i).eq.0) fout(6,i)=2.*f(6,jnear)
         if(f(1,i).eq.0) fout(7,i)=2.*f(7,jnear)
         if(f(1,i).eq.0) fout(8,i)=2.*f(8,jnear)
      enddo
      
      return
      end
