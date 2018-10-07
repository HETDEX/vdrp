
      parameter(iarrm=8000,arr2=10*iarrm)

      real xd(iarrm,iarrm),xp2(iarrm)
      real xt(iarrm),xr(iarrm),xp(iarrm),yl(iarrm),y2(iarrm)
      real xl(2),yl2(2)
      integer naxes(2)
      character im*80
      logical simple,extend,anyf

      data big/1.e20/

 1    call qc1('Image ','ima.def',im)
      call savdef

      nfile=0
      do i=1,30
c         if(im(i:i).eq.'.'.or.im(i:i).eq.' ') goto 966
         if(im(i:i).eq.' ') goto 966
         nfile=nfile+1
      enddo
 966  continue

      im1=0
      istatus=0
      call ftgiou(im1,istatus)
      iread=0
      call ftopen(im1,im,iread,iblock,ier)
      if(ier.ne.0) then
         print *,istatus,ier
         write(*,*) 'Error opening image : ',im
         goto 1
      endif

      call qi1('Which extension ','ima.def',nf)
      call ftmahd(im1,nf,ihd,ier)

      call ftghpr(im1,2,simple,ibit,naxis,naxes,ipc,igc,
     $     extend,ier)
      if(naxes(1).gt.iarrm.or.naxes(2).gt.iarrm) then
         write(*,"('Arrays too big - make iarrm bigger')")
         goto 706
      endif
      ncol=naxes(1)
      nrow=naxes(2)

      print *,ncol,nrow,ier
      call ftg2de(im1,igc,0.,iarrm,ncol,nrow,xd,anyf,ier)
      call ftclos(im1,ier)

 2    call qi1('Average rows(1) or cols(2) ','ima.def',ic)
      if(ic.ne.2.and.ic.ne.1) goto 2
      call qi2('Region ','ima.def',ir1,ir2)
      call qi2('Region2 ','ima.def',ir1b,ir2b)
      call savdef

      nr=0
      if(ic.eq.2) then
         do j=1,nrow
            n=0
            if(j.ge.ir1.and.j.le.ir2) then
               nr=nr+1
               xp(nr)=j
               sum=0.
               do i=max(1,ir1b),min(ncol,ir2b)
                  x=xd(i,j)
                  if(nint(x).ne.-666) then
                     n=n+1
                     xt(n)=x
                     sum=sum+x
                  endif
               enddo
               if(n.lt.1) then
                  nr=nr-1
                  goto 855
               endif
               call biwgt(xt,n,xb,xs)
               xr(nr)=xb
c               xr(nr)=sum/float(nt)
c               print *,nr,n,xr(nr)
            endif
 855        continue
         enddo
      elseif(ic.eq.1) then
         do i=1,ncol
            n=0
            if(i.ge.ir1.and.i.le.ir2) then
               nr=nr+1
               xp(nr)=i
               sum=0.
               do j=max(1,ir1b),min(nrow,ir2b)
                  x=xd(i,j)
                  if(nint(x).ne.-666) then
                     n=n+1
                     xt(n)=x
                     sum=sum+x
                  endif
               enddo
               if(n.lt.1) then
                  nr=nr-1
                  goto 856
               endif
               call biwgt(xt,n,xb,xs)
               xr(nr)=xb
c               xr(nr)=sum/float(nt)
            endif
 856        continue
         enddo
      endif

      ymax=-big
      ymin=big
      do i=1,nr
         if(xr(i).gt.ymax) then
            ymax=xr(i)
            xloc=xp(i)
         endif
         ymin=min(ymin,xr(i))
      enddo
         
      open(unit=11,file='imab.out',status='unknown')
      write(11,*) im(1:nfile),xloc
      close(11)
      print *
      print *,im(1:nfile),nf,xloc
      print *
      ybit=(ymax-ymin)/10.
      ymax=ymax+ybit
      ymin=ymin-ybit
      xl(1)=xloc
      xl(2)=xloc
      yl2(1)=ymin
      yl2(2)=ymax

      open(unit=11,file='ima.out',status='unknown')
      do i=1,nr
         write(11,*) xp(i),xr(i)
      enddo
      close(11)

      call pgbegin(0,'?',1,1)
      call pgpap(0.,1.)
      call pgscf(2)
      call pgsch(1.3)
      call pgslw(2)

c      ymin=0.7
c      ymax=0.8
c      ymin=0.
c      ymax=2e4
c      ymin=1e4
c      ymax=3e4
      call pgenv(xp(1),xp(nr),ymin,ymax,0,0)
c      if(ic.eq.2) call pglabel('Row','',im)
c      if(ic.eq.1) call pglabel('Col','',im)
      if(ic.eq.2) call pglabel('Row','','')
      if(ic.eq.1) call pglabel('Col','','')
      call pgsch(.5)
c      call pgpoint(nr,xp,xr,17)
      call pgline(nr,xp,xr)
c      call pgline(np,xp2,yl)
c      call pgline(2,xl,yl2)

      call pgend

 706  continue

      end
