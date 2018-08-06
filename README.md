# vdrp
HET VIRUS data reduction pipeline


## Installation
### Astrometry
Go to your run

  /work/?????/USER/maverick/sci/panacea/shifts

directory.

Run 

  git clone https://github.com/mxhf/vdrp.git

Link the runall1.sh, test.sh and setup.sh files.

  ln -s vdrp/runall1.sh runall1
  cp vdrp/setup.sh setup.sh
  ln -s vdrp/test.sh test.sh

Note, I omitted the extension to runall1 such that it serves as direct 
drop in replacement to the legacy runall1.

Copy the config file from the vdrp config subdirectory.

  cp vdrp/config/vdrp.config vdrp.config

Now edit and adjust setup.sh and vdrp.config to point to the 
correct paths.

You might want to runs the tests now.
  ./test.sh

Execute like
  runall1 20180611 017 13.8447 51.3479 1

Open runall1.sh for various options for executing the astrometry.py routine.


Proposed directory structure
.
├── daoprogs
│   ├── moreprogs
│   └── moreprogs2
│       └── dao2
│
├── sciprogs
│
├── vdrp
│   ├── bin
│   ├── config
│   ├── lib
│   └── src
│       ├── legacy
│       │   ├── astrometry
│       │   │   └── scripts
│       │   └── getcoords
│       └── python
│           ├── jynp
│           ├── tests
│           │   └── testdata
│           ├── validation
│           └── vdrp
│
│
├── vdrp.config or 20180611v017.vdrpconfig
├── setup.sh
│
├── reductions
│   └── 20180611
│       └── virus
│           └── virus0000017
│               ├── exp01
│               │   └── virus
│               ├── exp02
│               │   └── virus
│               └── exp03
│                   └── virus
├── shifts
│   ├── run_shifts.sh
│   └── 20180611v017
│
├── fibcoords
│   ├── run_fibcoords.sh
│   └── 20180611v017
│       ├── coords
│       ├── exp01
│       │   └── virus
│       ├── exp02
│       │   └── virus
│       └── exp03
│           └── virus
├── detect
│   ├── run_detect.sh

