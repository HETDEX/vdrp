import sys
import warnings

# if setup tools is not installed, bootstrap it
try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.sdist import sdist

import subprocess as sp

if sys.version_info < (2, 7) or sys.version_info >= (3, 0):
    sys.exit("Python version 2.7 required")


def run_install():
    src_path = './src/fortran'
    cmd = 'make pkg_install'
    sp.check_call(cmd, cwd=src_path, shell=True)


def run_make_clean():
    src_path = './src/fortran'
    cmd = 'make pkg_clean'
    sp.check_call(cmd, cwd=src_path, shell=True)


class BuildBinaries(build_ext):
    """Custom handler for the 'install' command."""
    def run(self):
        run_install()


class CustomBuild(build_py):
    def run(self):
        run_install()
        build_py.run(self)


class CustomInstall(install):
    def run(self):
        run_install()
        install.run(self)


class CustomDevelop(develop):
    def run(self):
        run_install()
        develop.run(self)


class CustomSDist(sdist):
    def run(self):
        run_make_clean()
        sdist.run(self)


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
                    'hetdex-shuffle', 'hetdex_vdrp_support', 'stellarSEDfits',
                    'path.py', 'paramiko', 'urllib3<1.24']

# entry points
# scripts
entry_points = {'console_scripts':
                ['vdrp_astrom = vdrp.astrometry:run',
                 'vdrp_throughput = vdrp.photometry:run',
                 'vdrp_calc_flim = vdrp.fluxlim:calc_fluxlim_entrypoint',
                 'vdrp_setup_flim = vdrp.fluxlim:setup_fluxlim_entrypoint',
                 'vdrp_bindir = vdrp.utils:print_bindir',
                 'vdrp_configdir = vdrp.utils:print_configdir',
                 'vdrp_config = vdrp.utils:print_conffile',
                 'vdrp_runner = vdrp.vdrprunner:run',
                 'vdrp_jobsplitter = vdrp.jobsplitter:run']}

# entry_points.update(batch_types)

setup(
    # package description and version
    name="vdrp",
    version='1.0.5',
    author="HETDEX collaboration",
    author_email="snigula@mpe.mpg.de",
    description="VIRUS data reduction pipeline",
    long_description=open("README.md").read(),

    # custom test class

    # list of packages and data
    package_dir={'': 'src', 'config': 'src/vdrp/config'},
    packages=find_packages('src', exclude=["pytest", "tests"]),
    package_data={'bin': ['vdrp/bin'], 'config': ['vdrp/config']},
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
                 ],
    cmdclass={'build_ext': BuildBinaries,
              'install': CustomInstall,
              'develop': CustomDevelop,
              'sdist': CustomSDist}
    # cmdclass={'develop': CustomDevelop}
)
