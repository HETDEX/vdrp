import threading
import multiprocessing.pool
import Queue
import time


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
        while True:
            try:
                func, args, kargs = self.tasks.get(True, 2.0)
                try:
                    func(*args, **kargs)
                except Exception as e:
                    print(e)
                finally:
                    self.tasks.task_done()
            except Queue.Empty:
                print('%s %s queue is empty, shutting down!'
                      % (time.strftime('%H:%M:%S'), self.name))
                return
            except Exception as e:
                print(e)


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
                func, args, kargs = self.tasks.get(True, 2.0)
                try:
                    func(*args, **kargs)
                except Exception as e:
                    print(e)
                finally:
                    self.tasks.task_done()
            except Queue.Empty:
                print('%s %s queue is empty, shutting down!'
                      % (time.strftime('%H:%M:%S'), self.name))
                return
            except Exception as e:
                print(e)


class ThreadPool:
    """Pool of threads consuming tasks from a queue"""
    def __init__(self, num_threads):
        self.tasks = Queue.Queue(num_threads)
        for i in range(num_threads):
            ThreadWorker('ThreadWorker%d' % i, self.tasks)

    def add_task(self, func, *args, **kargs):
        """Add a task to the queue"""
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        """Wait for completion of all the tasks in the queue"""
        self.tasks.join()


class MPPool:
    """Pool of threads consuming tasks from a queue"""
    def __init__(self, jobnum, num_proc):
        self.tasks = multiprocessing.JoinableQueue(num_proc)
        for i in range(num_proc):
            MPWorker('MPWorker%d_%d' % (jobnum, i), self.tasks)

    def add_task(self, func, *args, **kargs):
        """Add a task to the queue"""
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        """Wait for completion of all the tasks in the queue"""
        self.tasks.join()
