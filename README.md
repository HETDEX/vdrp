# vdrp
HET VIRUS data reduction pipeline


## Installation
Go to your data reduction directory e.g.

  cd /work/?????/USER/maverick/sci/panacea


Clone the GIT repository 
  git clone https://github.com/mxhf/vdrp.git


create the directories, shifts, specphot
    mkdir shifts, specphot


Link the config file.
  ln -s vdrp/config/vdrp.config .


In shifts and specphot, link the corresponding run scripts.
  cd shifts 
  ln -s ../vdrp/src/shifts/run_shifts.sh
  cd ..

In shifts and specphot, link the corresponding run scripts.
  cd specphot
  ln -s ../vdrp/src/shifts/run_specphot.sh
  cd ..


Execution
  # Astrometry
  cd shifts
  ./run_shifts.sh 20180611 017
  cd ..
  # alternatively give specific RA/Dec and track information
  ./run_shifts.sh 20180611 017 13.8447 51.3479 1

  cd specphot
  ./run_specphot.sh 20180611 017
  cd ..


