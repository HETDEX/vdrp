
      parameter (narrm=5000)
      real xd(narrm,narrm),xd2(narrm,narrm),xin(narrm),yin(narrm)
      real xs(narrm),ys(narrm),y3(narrm),xin2(narrm),yin2(narrm)
      real xin3(narrm),yin3(narrm)
      integer ibad(narrm),ibad2(narrm),naxes(2)
      character file1*40
      logical simple,extend,anyf

      ic306=0
      iflag=0
      nbin=1
c      val1=1e6
c      val2=1e6
      val1=0
      val2=0
      ratlo=0.96
      rathi=1.02
      ratlo=0.92
      rathi=1.06
      ibadr=5

 1    call qc1('Image ','imbox.def',file1)
      call qi1('Which extension ','imbox.def',iext)
      call qr1('Inner radius to ignore ','imbox.def',rad)
      call savdef

      im1=0
      ier=0
      call ftgiou(im1,ier)
      iread=0
      call ftopen(im1,file1,iread,iblock,ier)
      if(ier.ne.0) then
         write(*,*) 'Error opening image : ',file1
         goto 1
      endif
      call ftmahd(im1,iext,ihd,ier)
      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,extend,ier)
      if(naxes(1).gt.narrm.or.naxes(2).gt.narrm) then
         write(*,"('Arrays too small - make narrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=naxes(2)
      call ftg2de(im1,igc,0.,narrm,ncol,nrow,xd,anyf,ier)

      do j=1,nrow
c         if((float(j)/10.-int(j/10)).eq.0) 
c    $        write(6,"('Doing ',i4,a1,$)") j,char(13)
c         call flush(6)
         do i=1,ncol
            xin(i)=float(i)
            yin(i)=xd(i,j)
            xs(i)=float(i)
         enddo

         nin2=0
         do i=1,ncol,nbin
            nin2=nin2+1
            ymax=-1e10
            js=i
            je=i+nbin-1
            do jc=js,je
               if(yin(jc).gt.ymax) then
                  xmax=xin(jc)
                  ymax=yin(jc)
               endif
            enddo
            xin2(nin2)=xmax
            yin2(nin2)=ymax
         enddo
         nin3=0
         do i=1,nin2
            if(ic306.eq.1) then
               if(xin2(i).gt.280.and.xin2(i).lt.330) goto 306
            endif
            if(yin2(i).gt.0.001) then
               nin3=nin3+1
               xin3(nin3)=xin2(i)
               yin3(nin3)=yin2(i)
            endif
 306        continue
         enddo

         if(nin3.gt.10) then
            call smooth(nin3,xin3,yin3,ncol,xs,ys,y3,val1)
         else
            do i=1,ncol
               ys(i)=1.0
            enddo
         endif

         nin=0
         nini=0
         do i=1,nin3
            rat=yin3(i)/y3(i)
            if(rat.gt.ratlo.and.rat.lt.rathi) then
               nin=nin+1
               xin2(nin)=xin3(i)
               yin2(nin)=yin3(i)
            endif
            if(rat.le.ratlo.or.rat.ge.rathi) then
               nini=nini+1
               ibad(nini)=i
            endif
            ibad2(i)=0
         enddo
         do i=1,nini
            ilo=max(1,ibad(i)-ibadr)
            ihi=min(nin,ibad(i)+ibadr)
            do ib=ilo,ihi
               ibad2(ib)=1
            enddo
         enddo
         nin=0
         do i=1,nin3
            if(ic306.eq.1) then
               if(xin(i).gt.280.and.xin(i).lt.330) goto 307
            endif
            if(ibad2(i).eq.0) then
               nin=nin+1
               xin2(nin)=xin(i)
               yin2(nin)=yin(i)
            endif
 307        continue
         enddo

         if(nin.gt.10) then
            call smooth(nin,xin2,yin2,ncol,xs,ys,y3,val2)
         else
            do i=1,ncol
               ys(i)=1.0
            enddo
         endif

         do i=1,ncol
            xd2(i,j)=ys(i)
         enddo
      enddo

      ier=0
      call ftclos(im1,ier)
      call ftinit(51,'imbox.fits',iblock,ier)
      call ftphps(51,-32,naxis,naxes,ier)
      if(ier.ne.0) then
         print *,'Error in output file ',ier
         goto 706
      endif
      call ftp2de(51,igc,narrm,naxes(1),naxes(2),xd2,ier)
      call ftclos(51,ier)

 706  continue
      end

      subroutine smooth(n,x,y,n2,x2,y2,y3,valin)
      parameter(nmax=2000,mm=2,nwk=nmax+6*(nmax*mm+1),mm2=mm*2)
      real x(n),y(n),x2(n2),y2(n2),y3(n)
      real*8 dx(nmax),dy(nmax),wx(nmax),cf(nmax),wk(nwk),val,splder
      real*8 q(mm2)

c      call qd1('Enter smoothing val ','smflat.def',val)
c      call savdef
c      val=0.d0
      val=dble(valin)
c      val=1.d-1
      md=3
      if(val.eq.0.) md=2
      m=2

      do i=1,n
         dx(i)=dble(x(i))
         dy(i)=dble(y(i))
         wx(i)=1.d0
      enddo

      call gcvspl(dx,dy,nmax,wx,1.d0,m,n,1,md,val,cf,nmax,wk,ier)
      if(ier.ne.0) print *,'ier= ',ier

      do i=1,n2
         in=i
         y2(i)=sngl(splder(0,m,n,dble(x2(i)),dx,cf,in,q))
      enddo

      do i=1,n
         in=i
         y3(i)=sngl(splder(0,m,n,dble(x(i)),dx,cf,in,q))
      enddo

      return
      end
