
      real fm(6),bm(6),fmn(6)
      character n1*8,n2*3,nbase*12,nd(6)*10

      read *,n1,n2
      nbase=n1//"v"//n2

      nd(1)='e01gp1.mof'
      nd(2)='e02gp1.mof'
      nd(3)='e03gp1.mof'
      nd(4)='e01gp2.mof'
      nd(5)='e02gp2.mof'
      nd(6)='e03gp2.mof'
      n=0
      do i=1,6
         open(unit=1,file="all/"//nbase//nd(i),status='old',err=667)
         read(1,*) x1,x2,x3
         if(x2.gt.0.5.and.x2.lt.4.0) then
            if(x3.lt.1.0) x3=3.9
            if(x3.gt.10.0) x3=9.9
            n=n+1
            fm(n)=x2
            bm(n)=x3
         endif
 667     continue
      enddo
      close(1)
      if(n.ge.1) then
         call biwgt(fm,n,xbf,xs)
         call biwgt(bm,n,xbb,xs)
      else
         xbf=1.666
         xbb=3.9
      endif

      x11=1
      x12=1
      x13=1
      x21=1
      x22=1
      x23=1
      open(unit=1,file="all/"//nbase//".nm1",status='old',err=668)
      read(1,*,end=668,err=668) x11,x12,x13
 668  continue
      close(1)
      open(unit=1,file="all/"//nbase//".nm2",status='old',err=670)
      read(1,*,end=670,err=670) x21,x22,x23
 670  continue
      close(1)
      x11=max(0.,x11)
      x12=max(0.,x12)
      x13=max(0.,x13)
      x21=max(0.,x21)
      x22=max(0.,x22)
      x23=max(0.,x23)

      open(unit=1,file="all/"//nbase//"gp1.comp",status='old')
      read(1,*) fm(1),fm(2),fm(3)
      close(1)
      open(unit=1,file="all/"//nbase//"gp2.comp",status='old')
      read(1,*) fm(4),fm(5),fm(6)
      close(1)
      nn=0
      do i=1,6
         if(fm(i).gt.0.5.and.fm(i).lt.4.0) then
            nn=nn+1
            fmn(nn)=fm(i)
         endif
      enddo
      call biwgt(fmn,nn,xf,xs)
      if(nn.lt.1) xf=1.666

      open(unit=11,file='out',status='unknown')
      write(11,1001) nbase,"Dex",xf,xbf,xbb,n,x11,x12,x13,x21,x22,x23
      close(11)
 1001 format(a12,1x,a3,3(1x,f6.3),1x,i2,6(1x,f6.3))

      end
