
      parameter(nmax=10000)
      real x(nmax),y(nmax),ye(nmax),ya(nmax,200),xin(200)
      real yel(nmax),yeu(nmax),ydiff(nmax),ydiffn(nmax)
      real yorig(nmax)
      integer ica(nmax)
      character file1*80,file2*80,lista(4)*6,c1*3

      lista(1)="listLL"
      lista(2)="listLU"
      lista(3)="listRL"
      lista(4)="listRU"
      open(unit=1,file='title',status='old')
      read(1,*) c1
      close(1)

      call pgbegin(0,'?',2,2)
      call pgpap(0.,1.)
      call pgsch(1.)
      call pgscf(2)
      call pgslw(2)

      xmin=3500.
      xmax=5450.
      ymin=0.6
      ymax=1.5

      do ia=1,4
         open(unit=1,file=lista(ia),status='old')
         call pgsci(1)
         call pgenv(xmin,xmax,ymin,ymax,0,0)
         call pglabel('Wavelength','FtF','')
         call pgsch(2.5)
         call pgmtxt('T',-1.5,0.5,0.5,c1)
         if(ia.eq.1) call pgmtxt('T',-1.5,0.64,0.5,"LL")
         if(ia.eq.2) call pgmtxt('T',-1.5,0.64,0.5,"LU")
         if(ia.eq.3) call pgmtxt('T',-1.5,0.64,0.5,"RL")
         if(ia.eq.4) call pgmtxt('T',-1.5,0.64,0.5,"RU")
         call pgsch(1.0)
         call pgsls(1)
         call pgslw(1)

         nl=0
         do il=1,1000
            read(1,*,end=666) file1,ic
            open(unit=2,file=file1,status='old')
            nl=nl+1
            n=0
            do i=1,2000
               read(2,*,end=667) x1,x2
               n=n+1
               x(n)=x1
               y(n)=x2
               ya(n,il)=x2
            enddo
 667        continue
            close(2)
            ica(il)=ic
         enddo
 666     continue
         close(1)
         
         do i=1,n
            do j=1,nl
               xin(j)=ya(i,j)
            enddo
            call biwgt(xin,nl,xb,xs)
            y(i)=xb
            yel(i)=y(i)-xs/sqrt(float(nl))
            yeu(i)=y(i)+xs/sqrt(float(nl))
         enddo

         fac=1.0
         do i=1,n
            y(i)=y(i)*fac
         enddo

         do j=1,nl
            do i=1,n
               ydiff(i)=y(i)/ya(i,j)
               yorig(i)=ya(i,j)
            enddo
            call biwgt(ydiff,n,xb,xs)
            do i=1,n
               ydiff(i)=ya(i,j)*xb
            enddo
            do i=1,n
               ydiffn(i)=ydiff(i)-y(i)
            enddo
            call biwgt(ydiffn,n,xb,xs)
            if(xs.gt.0.0) then
               ic=ic+1
               if(ic.eq.15) ic=1
               ic=ica(j)
               call pgsci(ic)
               call pgline(n,x,yorig)
            endif
         enddo
      enddo

      call pgend

      end
