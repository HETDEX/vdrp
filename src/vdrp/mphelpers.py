import threading
import multiprocessing.pool
import queue
import time
import logging
import os
import sys
import copy

_logger = logging.getLogger()


# Parallelization code, we supply both a ThreadPool as well as a
# multiprocessing pool. Both start a given numer of threads/processes,
# that will work through the supplied tasks, till all are finished.
#
# The ThreadPool does not need to start subprocesses, but is limited by
# the Python Global Interpreter Lock (only one thread can access complex data
# types at one time). This can potentially slow things down.
#
# The MPPool needs to start up the processes, but this is only done once at
# the initializtion of the pool.
#
# The MPPool processes cannot start multiprocessing jobs themselves, so if
# you need nested parallelization, use the either ThreadPools for all, or
# Use one and the other.


class ThreadShutDownException():

    pass


def shutdownThread():

    raise ThreadShutDownException()


class ThreadWorker(threading.Thread):
    """Thread executing tasks from a given tasks queue"""
    def __init__(self, name, tasks):
        threading.Thread.__init__(self)
        self.name = name
        self.tasks = tasks
        self.daemon = True
        self.start()

    def run(self):
        threading.current_thread().name = self.name
        _logger.debug('Starting Thread %s' % self.name)
        while True:
            try:
                func, args, kargs = self.tasks.get(True, 120.0)
                _logger.debug('Got new task from queue')
                _logger.debug('There are approx. %d tasks waiting'
                              % self.tasks.qsize())
                try:
                    func(*args, **kargs)
                except ThreadShutDownException:
                    _logger.info('Shutting down thread')
                    break
                except Exception as e:
                    _logger.exception(e)
                finally:
                    self.tasks.task_done()
            except queue.Empty:
                print('%s %s queue is empty, shutting down!'
                      % (time.strftime('%H:%M:%S'), self.name))
                return
            except Exception as e:
                _logger.exception(e)


class MPWorker(multiprocessing.Process):
    """Thread executing tasks from a given tasks queue"""
    def __init__(self, name, tasks):
        multiprocessing.Process.__init__(self)
        self.name = name
        self.tasks = tasks
        self.start()

    def run(self):
        while True:
            try:
                func, args, kargs = self.tasks.get(True, 1200.0)
                _logger.debug('Got new task from queue')
                _logger.debug('There are approx. %d tasks waiting'
                              % self.tasks.qsize())
                try:
                    func(*args, **kargs)
                except ThreadShutDownException:
                    _logger.info('Shutting down thread')
                    break
                except Exception as e:
                    _logger.exception(e)
                finally:
                    self.tasks.task_done()
            except queue.Empty:
                print('%s %s queue is empty, shutting down!'
                      % (time.strftime('%H:%M:%S'), self.name))
                return
            except Exception as e:
                _logger.exception(e)


class ThreadPool:
    """Pool of threads consuming tasks from a queue"""
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.tasks = queue.Queue()
        for i in range(num_threads):
            ThreadWorker('ThreadWorker%d' % i, self.tasks)

    def add_task(self, func, *args, **kargs):
        """Add a task to the queue"""
        _logger.info('Adding new task to queue')
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        """Wait for completion of all the tasks in the queue"""
        _logger.info('Job submission complete, adding shutdown jobs')
        for i in range(self.num_threads):
            self.add_task(shutdownThread)
        self.tasks.join()


class MPPool:
    """Pool of threads consuming tasks from a queue"""
    def __init__(self, jobnum, num_proc):
        self.num_proc = num_proc
        self.tasks = multiprocessing.JoinableQueue()
        for i in range(num_proc):
            print('Creating mp workers')
            MPWorker('MPWorker%d_%d' % (jobnum, i), self.tasks)

    def add_task(self, func, *args, **kargs):
        """Add a task to the queue"""
        _logger.info('Adding new task %s to queue' % func)
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        """Wait for completion of all the tasks in the queue"""
        _logger.info('Job submission complete, adding shutdown jobs')
        for i in range(self.num_proc):
            self.add_task(shutdownThread)
        self.tasks.join()


def mp_run(func, args, rargv, parser):

    # We found a -M flag with a command file, now loop over it, we parse
    # the command line parameters for each call, and intialize the args
    # namespace for this call.
    if args.multi:
        mfile = args.multi.split('[')[0]

        if not os.path.isfile(mfile):
            raise Exception('%s is not a file?' % mfile)

        try:  # Try to read the file
            with open(mfile) as f:
                cmdlines = f.readlines()
        except Exception as e:
            _logger.exception(e)
            raise Exception('Failed to read input file %s!' % args.multi)

        # Parse the line numbers to evaluate, if any given.
        if args.multi.find('[') != -1:
            try:
                minl, maxl = args.multi.split('[')[1].split(']')[0].split(':')
            except ValueError:
                raise Exception('Failed to parse line range, should be of '
                                'form [min:max]!')

            cmdlines = cmdlines[int(minl):int(maxl)]

        # Create the ThreadPool.
        pool = ThreadPool(args.mcores)
        c = 1

        # For each command line add an entry to the ThreadPool.
        for l in cmdlines:
            largs = copy.copy(rargv)
            largs += l.split()

            main_args = parser(largs)

            pool.add_task(func, c, copy.copy(main_args))

        # Wait for all tasks to complete
        pool.wait_completion()

        sys.exit(0)
    else:
        # Parse config file and command line paramters
        # command line parameters overwrite config file.

        # The first positional argument wasn't an input list,
        # so process normally
        args = parser(rargv)

        sys.exit(func(1, args))
