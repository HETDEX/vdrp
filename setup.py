import sys
import warnings

# if setup tools is not installed, bootstrap it
try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import setup, find_packages

if sys.version_info < (2, 7) or sys.version_info >= (3, 0):
    sys.exit("Python version 2.7 required")


def extras_require(key=None):
    """Deal with extra requirements

    Parameters
    ----------
    key : string, optional
        if not none, returns the requirements only for the given option

    Returns
    -------
    dictionary of requirements
    if key is not None: list of requirements
    """
    req_dic = {}
    req_dic = {'doc': ['sphinx>=1.4', 'numpydoc>=0.6', 'alabaster',
                       'sphinxcontrib-httpdomain', ]
               }

    req_dic['livedoc'] = req_dic['doc'] + ['sphinx-autobuild>=0.5.2', ]

    req_dic['test'] = ['robotframework', 'robotframework-requests',
                       'pytest-cov', 'coverage>=4.2']
    req_dic['tox'] = ['tox', 'tox-pyenv']

    req_dic['all'] = set(sum((v for v in req_dic.values()), []))

    if key:
        return req_dic[key]
    else:
        return req_dic


install_requires = ['numpy', 'scipy', 'matplotlib', 'astropy', 'pyhetdex',
                    'stellarSEDfits', 'path.py']

# entry points
# scripts
entry_points = {'console_scripts':
                ['vdrp_astrom = vdrp.astrometry:run',
                 'vdrp_throughput = vdrp.photometry:run']}

# entry_points.update(batch_types)

setup(
    # package description and version
    name="vdrp",
    version='0.4.0',
    author="HETDEX collaboration",
    author_email="snigula@mpe.mpg.de",
    description="VIRUS data reduction pipeline",
    long_description=open("README.md").read(),

    # custom test class

    # list of packages and data
    package_dir={'': 'src'},
    packages=find_packages('src', exclude=["pytest", "tests"]),
    # get from the MANIFEST.in file which extra file to include
    include_package_data=True,
    # don't zip when installing
    zip_safe=False,

    # entry points: creates vhc script upon installation
    entry_points=entry_points,
    # dependences
    # setup_requires=['pytest-runner', ],
    install_requires=install_requires,
    extras_require=extras_require(),

    classifiers=["Development Status :: 3 - Alpha",
                 "Environment :: Console",
                 "Intended Audience :: Science/Research",
                 "License :: OSI Approved :: GNU General Public License (GPL)",
                 "Operating System :: POSIX :: Linux",
                 "Programming Language :: Python :: 2.7",
                 "Topic :: Scientific/Engineering :: Astronomy",
                 ]
)
