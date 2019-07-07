
      parameter(nmax=100)
      parameter(radtodeg=57.29578)
      real r(nmax),v(nmax),rmid(nmax),vmid(nmax)
      character fname*20,cvel*30

      rmax=1.
      rmax=400.
      rmax=1.2
      slitsize=4.
      open(unit=1,file='bin_r.out',status='old')
      read(1,*)
      read(1,*)
      nr=0
      do i=1,nmax
         read(1,*,end=666) i1,x2,x3,x4
         nr=nr+1
         r(nr)=x4
         rmid(nr)=x3
         irmax=i1
      enddo
 666  continue
      close(1)
      open(unit=1,file='bin_v.out',status='old')
      read(1,*)
      read(1,*)
      nv=0
      do i=1,nmax
         read(1,*,end=667) i1,x2,x3,x4
         nv=nv+1
         v(nv)=x4
         vmid(nv)=x3
         ivmax=i1
      enddo
 667  continue
      close(1)

      write(*,"('Input rotation axis : '$)")
      read *,ang

      call pgbegin(0,'?',1,1)
      call pgscf(2)
      call pgslw(2)
      call pgsch(1.5)

      call pgenv(0.,rmax,0.,rmax,1,0)
      call drawcirc(nr,r)
      call drawang(nv,v,r(1),r(nr))
c      call drawslit(slitsize,2.0)

      open(unit=1,file='binvr.dat',status='old',err=689)
      read(1,*)
      do i=1,100
         read(1,*,end=689) vlo,vup,rlo,rup,ic
         call pgsci(ic)
         call pgslw(5)
c         call drawang1(vlo,vup,rlo,rup)
      enddo
 689  continue
      close(1)
      call pgsci(1)
      call pgslw(2)
      call pgsch(1.0)
      call pgsch(0.5)

      open(1,file='vels.dat',status='old')
      open(12,file='names.out',status='unknown')
      sa=sin(ang/radtodeg)
      ca=cos(ang/radtodeg)
 2001 format(2(1x,f8.3))
      do i=1,100000
         read(1,*,end=668) xp,yp,cvel
c         read(1,*,end=668) xp,yp,cvel,dvel
c         write(cvel,2001) vel,dvel
         x=xp*ca-yp*sa
         y=xp*sa+yp*ca
c         print *,x,y
         ivel=1
         dvel=1
         call pgsci(1)
         if(x.lt.0) then
            x=-x
            ivel=0
            call pgsci(2)
            vel=-vel
            dvel=-1
c -- make ivel=1 to not have bnn's, =0 to have them
            ivel=0
         endif
         itype=17
         if(y.lt.0) itype=23
         y=abs(y)
         call pgpt1(x,y,itype)
         call pgsci(1)
         rp=sqrt(x*x+y*y)
         if(abs(x).ne.0) then
            vp=radtodeg*atan(y/x)
         else
            vp=89.99
         endif
         vp=min(vp,89.99)
         ir=-1
         iv=-1
         do j=2,irmax
            if(rp.ge.r(j-1).and.rp.lt.r(j)) ir=j
            if(rp.lt.r(1)) ir=1
         enddo               
         do j=2,ivmax
            if(vp.ge.v(j-1).and.vp.lt.v(j)) iv=j
            if(vp.lt.v(1)) iv=1
         enddo               
         if(iv.lt.1) print *,iv,vp,radtodeg*atan(y/x)
         call writebin(ir,iv,cvel,fname,i,ivel,dvel)
         xb=rmid(ir)*cos(vmid(iv)/radtodeg)
         yb=rmid(ir)*sin(vmid(iv)/radtodeg)
         write(12,1201) fname(1:7),xb,yb
      enddo
 668  continue
      close(1)
      close(12)
 1201 format(a7,2(1x,f8.3))

      call pgend
      end

      subroutine writebin(ir,iv,cvel,fname,i,ivel,dvel)
      character fname*20,cvel*30
      if(ir.lt.1.or.iv.lt.1) then
         print *,'Not in Array',ir,iv,i,cvel
         return
      endif
      fname(1:3)='bin'
      if(ivel.eq.0) fname(1:3)='bnn'
      if(ir.lt.10) write(fname(4:5),1001) ir
      if(ir.ge.10) write(fname(4:5),1002) ir
      if(iv.lt.10) write(fname(6:7),1001) iv
      if(iv.ge.10) write(fname(6:7),1002) iv
      open(unit=11,file=fname(1:7),status='new',err=666)
      write(11,*) cvel,dvel
      goto 676
 666  continue
      close(11)
      open(unit=11,file=fname(1:7),status='old',access='append')
      write(11,*) cvel,dvel
 676  continue
      close(11)
 1001 format("0",i1)
 1002 format(i2)
      return
      end

      subroutine drawslit(y1,y2)
      real xl(10),xu(10)
      xoff=0.0
      nb=8
c- normal apertures
      xl(1)=-0.025
      xu(1)=0.025
      xl(2)=0.025
      xu(2)=0.075
      xl(3)=0.075
      xu(3)=0.125
      xl(4)=0.125
      xu(4)=0.225
      xl(5)=0.225
      xu(5)=0.375
      xl(6)=0.375
      xu(6)=0.625
      xl(7)=0.625
      xu(7)=0.975
      xl(8)=0.975
      xu(8)=1.525
c- new apertures
      xl(1)=0.
      xu(1)=2
      xl(2)=2
      xu(2)=4
      xl(3)=4
      xu(3)=6
      xl(4)=6
      xu(4)=8
      xl(5)=8
      xu(5)=10
      xl(6)=10
      xu(6)=12
      xl(7)=12
      xu(7)=14
      xl(8)=14
      xu(8)=16
      call pgsci(2)
      call pgsfs(2)
      do i=1,nb
         call pgrect(xl(i)+xoff,xu(i)+xoff,-y1/2.,+y1/2.)
      enddo
      call pgsci(1)
      return
      end

      subroutine drawcirc(n,r)
      parameter(radtodeg=57.29578)
      real r(n),xl(100),yl(100)
      xmin=0.
      do i=1,n
         rp=r(i)
         xmax=rp
         do j=1,100
            xl(j)=xmin+(xmax-xmin)*float(j-1)/99.
            xl(j)=min(rp,xl(j))
            yl(j)=sqrt(rp*rp-xl(j)*xl(j))
         enddo
         call pgline(100,xl,yl)
      enddo
      return
      end
      subroutine drawang(n,r,r1,rn)
      parameter(radtodeg=57.29578)
      real r(n),xl(100),yl(100)
      xmax=rn
      do i=1,n
         rp=r(i)
         xmin=r1*cos(rp/radtodeg)
         do j=1,100
            xl(j)=xmin+(xmax-xmin)*float(j-1)/99.
            yl(j)=xl(j)*tan(rp/radtodeg)
         enddo
         call pgline(100,xl,yl)
      enddo
      return
      end
      subroutine drawang1(vlo,vup,rlo,rup)
      parameter(radtodeg=57.29578)
      real xl(100),yl(100)
      xlo=rlo*cos(vlo/radtodeg)
      xup=rup*cos(vlo/radtodeg)
      do j=1,100
         xl(j)=xlo+(xup-xlo)*float(j-1)/99.
         yl(j)=xl(j)*tan(vlo/radtodeg)
      enddo
      call pgline(100,xl,yl)
      xlo=rlo*cos(vup/radtodeg)
      xup=rup*cos(vup/radtodeg)
      do j=1,100
         xl(j)=xlo+(xup-xlo)*float(j-1)/99.
         yl(j)=xl(j)*tan(vup/radtodeg)
      enddo
      call pgline(100,xl,yl)
      xlo=rlo*cos(vup/radtodeg)
      xup=rlo*cos(vlo/radtodeg)
      do j=1,100
         xl(j)=xlo+(xup-xlo)*float(j-1)/99.
         yl(j)=sqrt(rlo*rlo-xl(j)*xl(j))
      enddo
      call pgline(100,xl,yl)
      xlo=rup*cos(vup/radtodeg)
      xup=rup*cos(vlo/radtodeg)
      do j=1,100
         xl(j)=xlo+(xup-xlo)*float(j-1)/99.
         yl(j)=sqrt(rup*rup-xl(j)*xl(j))
      enddo
      call pgline(100,xl,yl)
      return
      end
