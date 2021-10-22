
subroutine grayscale(image,x,y,output) !return grayscale image
  !f2py intent(in) :: image,x,y
  !f2py intent(out) output

  integer :: x,y

  REAL :: image(1:x,1:y,1:3)

  REAL :: output(1:x,1:y)
  REAL :: R(1:x,1:y),G(1:x,1:y),B(1:x,1:y)
  REAL :: red,green,blue
  integer :: i,j


  R = image(:,:,1)
  G = image(:,:,2)
  B = image(:,:,3)
  do j = 1,y
    do i = 1,x
      red = R(i,j)*0.2989
      green = G(i,j)*0.5870
      blue = B(i,j)*0.114
      output(i,j) = red + blue + green
    end do
  end do

end subroutine grayscale


subroutine canny(h,l,image,x,y,output) !apply canny edge detection
  !f2py intent(in) :: image,x,y,h,l
  !f2py intent(out) :: output

  INTEGER :: x
  INTEGER :: y
  REAL :: image(1:x,1:y)
  REAL :: h,l
  Integer :: output(1:x,1:y)
  REAL :: pad(1:x+4,1:y+4)
  REAL :: conv(1:x,1:y)
  REAL :: convpad(1:x+1,1:y+1)
  REAL :: smallwindow(1:3,1:3)
  REAL :: fsmallwindow(1:9)
  REAL :: gfilter(1:5,1:5)
  REAL :: fgauss(1:25)
  REAL :: window(1:5,1:5)
  REAL :: fwindow(1:25)
  REAL :: point(1:25),point2(1:9),point3(1:9)
  REAL :: kx(1:3,1:3),ky(1:3,1:3)
  REAL :: fkx(1:9),fky(1:9)
  REAL :: filtx(1:x,1:y), filty(1:x,1:y)
  REAL :: IX,IY
  REAL :: G(1:x,1:y), theta(1:x,1:y)
  REAL :: PI = 3.1415926535897
  REAL :: angle
  REAL :: highThresh, lowThresh
  REAL :: thresh(1:x,1:y)

  INTEGER :: i,j,k,q,u
  REAL :: b1,b2


  !initialize filter
  gfilter(:,1) = (/ 2.,4.,5.,4.,2. /)
  gfilter(:,2) = (/ 4.,9.,12.,9.,4. /)
  gfilter(:,3) = (/ 5.,12.,15.,12.,5. /)
  gfilter(:,4) = (/ 4.,9.,12.,9.,4. /)
  gfilter(:,5) = (/ 2.,4.,5.,4.,2. /)

  do j = 1,5
    do i  = 1,5
      gfilter(i,j) = gfilter(i,j)/159.0
    end do
  end do
  fgauss = pack(gfilter, .TRUE.)
  !pad the image

  pad = 0.0
  do j = 3,y+2
    do i = 3,x+2
      pad(i,j) = image(i-2,j-2)
    end do
  end do


  !convolve


  do j = 3,y+2
    do i = 3,x+2

      q=-2
      do k = 1,5
        window(:,k) = pad(i-2:i+2,j+q)
        q = q+1
      end do

      fwindow = pack(window, .TRUE.)

      do u = 1,25
        point(u) = fwindow(u)*fgauss(u)
      end do

      conv(i-2,j-2) = sum(point)
    end do
  end do



  !gradient -> convolve again
  kx(:,1) = (/1,2,1/)
  kx(:,2) = (/0,0,0/)
  kx(:,3) = (/-1,-2,-1/)

  ky(:,1) = (/-1,0,1/)
  ky(:,2) = (/-2,0,2/)
  ky(:,3) = (/-1,0,1/)

  fkx = pack(kx, .TRUE.)
  fky = pack(ky, .TRUE.)


  convpad = 0.0
  do j = 2,y+1
    do i = 2,x+1
      convpad(i,j) = conv(i-1,j-1)
    end do
  end do



  do j = 2,y+1
    do i = 2,x+1
      q = -1
      do k = 1,3
        smallwindow(:,k) = convpad(i-1:i+1,j+q)
        q = q+1
      end do

      fsmallwindow = pack(smallwindow, .TRUE.)

      do u = 1,9
        point2(u) = fsmallwindow(u) * fkx(u)
        point3(u) = fsmallwindow(u) * fky(u)
      end do

      filtx(i-1,j-1) = sum(point2)
      filty(i-1,j-1) = sum(point3)

    end do
  end do





  do j = 1,y
    do i = 1,x
      IX = filtx(i,j)
      IY = filty(i,j)
      G(i,j) = sqrt((IX*IX) + (IY*IY))
      theta(i,j) = ATAN2(IY,IX) * 180./PI
      if (theta(i,j) < 0)THEN
        theta(i,j) = theta(i,j) + 180
      end if
    end do
  end do


  !non max suppression
  thresh = 0.0



  do j = 2,y-1
    do i = 2,x-1
      angle = theta(i,j)

      if (((angle >= 0) .and. (angle < 22.5)) .or. ((angle >= 15.*180./8.) .and. (angle <= 360.)))THEN
        b1 = G(i,j-1)
        b2 = G(i,j+1)

      else if ((angle >= 22.5) .and. (angle < 3.*180./8.).or.((angle >= 9.*180./8.).and.(angle<11.*180./8.)))THEN
        b1 = G(i+1,j-1)
        b2 = G(i-1,j+1)

      else if ((angle >= 3.*180./8.).and.(angle<5.*180./8.).or.((angle >= 11.*180./8.).and.(angle<13.*180./8.)))THEN
        b1 = G(i-1,j)
        b2 = G(i+1,j)

      else
        b1 = G(i-1,j-1)
        b2 = G(i+1,j+1)
      end if

      if ((G(i,j) >= b1) .and. (G(i,j) >= b2)) THEN
        thresh(i,j) = G(i,j)
      else
        thresh(i,j) = 0.0
      end if
    end do
  end do



  !Double Thresholding
  highThresh = h*255.
  lowThresh = l*255.



  do j = 1,y
    do i = 1,x
      if (thresh(i,j) >= highThresh) THEN
        thresh(i,j) = 255


      else if (thresh(i,j) < lowThresh) THEN
        thresh(i,j) = 0


      else if ((thresh(i,j) < highThresh) .and. (thresh(i,j) >= lowThresh)) THEN
        thresh(i,j) = lowThresh
      end if
    end do
  end do


  !hysteresis
  smallwindow = 0.0
  output = 0.0


  do j = 2,y-1
    do i = 2,x-1
      output(i,j) = INT(thresh(i,j)/255)
      if (thresh(i,j) == lowThresh)THEN
        q = -1
        do k = 1,3
          smallwindow(:,k) = thresh(i-1:i+1,j+q)
          q = q+1
        end do

        fsmallwindow = pack(smallwindow, .TRUE.)

        if (maxval(fsmallwindow) == 255) THEN
          output(i,j) = 1
        else
          output(i,j) = 0
        end if
      end if
    end do
  end do


end subroutine canny

subroutine hough(MD,image,x,y,output)
  !f2py intent(in) :: MD,image,x,y
  !f2py intent(out) :: output
  INTEGER :: x,y
  INTEGER :: image(1:x,1:y)
  REAL :: theta(1:180)
  INTEGER :: MD
  REAL :: output(1:2*MD,1:180)


  REAL :: PI = 3.1415926535897
  INTEGER :: j,i,k,r,t
  t = -90
  do k = 1,180
    theta(k) = t * PI/180.
    t = t+1
  end do

  output = 0.0
  do i = 1,x
    do j = 1,y
      if (image(i,j) > 0) THEN
        do k = 1,180
          r = (i-1)*COS(theta(k))+(j-1)*SIN(theta(k))
          output(int(r)+MD,k) = output(int(r)+MD,k)+1
        end do
      end if
    end do
  end do

end subroutine hough

!overlay subroutine
!overlay black portions onto grayscale image

subroutine overlay3(image,filt,x,y,output)
  !f2py intent(in) :: image,filt, x, y
  !f2py intent(out) :: output

  INTEGER :: x,y,k
  INTEGER :: image(1:x,1:y,1:3)
  INTEGER :: filt(1:x,1:y)
  INTEGER :: output(1:x,1:y,1:3)


  do j = 1,y
    do i = 1,x
      if (filt(i,j) == 1) THEN
        do k = 1,3
          output(i,j,k) = image(i,j,k)
        end do
      else
        do k = 1,3
          output(i,j,k) = 255!image(i,j,k)
        end do
      end if
    end do
  end do
end subroutine overlay3

subroutine overlay(filt,image,x,y,output)
  !f2py intent(in) :: image,filt, x, y
  !f2py intent(out) :: output
  INTEGER :: x,y
  REAL :: image(1:x,1:y),filt(1:x,1:y)
  REAL :: output(1:x,1:y)


  do j = 1,y
    do i = 1,x
      if (filt(i,j) == 1) THEN
        image(i,j) = 1
      end if
    end do
  end do
  output = image
end subroutine overlay

subroutine flip(image,x,y,output)
  !f2py intent(in):: image,x,y
  !f2py intent(out):: output
  INTEGER :: x,y
  INTEGER :: image(1:x,1:y)
  INTEGER :: output(1:x,1:y)
  INTEGER :: i,j

  do j = 1,y
    do i = 1,x
      if (image(i,j) == 0)THEN
        output(i,j) = 1
      else
        output(i,j) = 0
      end if
    end do
  end do

end subroutine flip

subroutine dilate(lim,image,x,y,output)
  !f2py intent(in) :: lim,image,x,y
  !f2py intent(out) :: output

  INTEGER :: x,y,lim
  INTEGER :: image(1:x,1:y),pad(1:x+4,1:y+4)
  INTEGER :: output(1:x,1:y)
  INTEGER :: i,j,k,q
  INTEGER :: window(1:5,1:5)
  INTEGER :: fwindow(1:25)

  pad = 1
  output = image

  do j = 3,y+2
    do i = 3,x+2
      pad(i,j) = image(i-2,j-2)
    end do
  end do



  do j = 3,y+2
    do i = 3,x+2

      q=-2
      do k = 1,5
        window(:,k) = pad(i-2:i+2,j+q)
        q = q+1
      end do

      fwindow = pack(window, .TRUE.)

      if (sum(fwindow) >= lim) THEN
        output(i-2,j-2) = 1
      end if
    end do
  end do

end subroutine dilate

subroutine fill(lim,image,x,y,output)
  !f2py intent(in) :: lim,image,x,y
  !f2py intent(out) :: output

  INTEGER :: x,y,lim
  INTEGER :: image(1:x,1:y),pad(1:x+4,1:y+4)
  INTEGER :: output(1:x,1:y)
  INTEGER :: i,j,k,q
  INTEGER :: window(1:5,1:5)
  INTEGER :: fwindow(1:25)

  pad = 1
  output = image

  do j = 3,y+2
    do i = 3,x+2
      pad(i,j) = image(i-2,j-2)
    end do
  end do

  if (pad(i,j) == 1)THEN


    do j = 3,y+2
      do i = 3,x+2

        q=-2
        do k = 1,5
          window(:,k) = pad(i-2:i+2,j+q)
          q = q+1
        end do

        fwindow = pack(window, .TRUE.)

        if (sum(fwindow) <= lim) THEN
          output(i-2,j-2) = 0
        end if
      end do
    end do


  end if

end subroutine fill


subroutine binarize(thresh,image,x,y,output)
  !f2py intent(in) :: thresh,image,x,y
  !f2py intent(out) :: output

  INTEGER :: x,y
  REAL :: image(1:x,1:y)
  REAL :: thresh
  INTEGER :: output(1:x,1:y)
  INTEGER :: i,j

  do j = 1,y
    do i = 1,x
      if (image(i,j) >= thresh)THEN
        output(i,j) = 1
      else
        output(i,j) = 0
      end if
    end do
  end do

end subroutine binarize
