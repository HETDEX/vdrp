from collections import OrderedDict
import os
import pickle
import subprocess
import logging

_logger = logging.getLogger()


class VdrpInfo(OrderedDict):
    def __init__(self, *args, **kwargs):
        super(VdrpInfo, self).__init__(*args, **kwargs)

    def save(self, dir, filename='vdrp_info.pickle'):
        # save arguments for the execution
        with open(os.path.join(dir, filename), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def read(cls, dir, filename='vdrp_info.pickle'):
        if os.path.exists(os.path.join(dir, filename)):
            with open(os.path.join(dir, filename), 'rb') as f:
                return pickle.load(f)
        else:
            return VdrpInfo()


def save_data(d, filename):
    # save data for later tasks
    with open(filename, 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)


def read_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def run_command(cmd, input=None, wdir=None):
    """
    Run and fortran command sending the optional input string on stdin.

    Parameters
    ----------
    cmd : str
        The command to be run, must be full path to executable
    input : str, optional
        Input to be sent to the command through stdin.
    """
    _logger.info('Running %s' % cmd)
    _logger.debug('Command params are %s' % input)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, cwd=wdir,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    so, _ = proc.communicate(input=input.encode())
    for l in so.split(b"\n"):
        _logger.info(l.decode())
    proc.wait()
