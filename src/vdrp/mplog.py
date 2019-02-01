# from logging.handlers import RotatingFileHandler
import multiprocessing
import threading
import logging
import sys
import traceback
import Queue
import logging

_logger = logging.getLogger()


def install_mp_handler(logger=None):
    """Wraps the handlers in the given Logger with an MultiProcessingHandler.
    :param logger: whose handlers to wrap. By default, the root logger.
    """
    if logger is None:
        logger = logging.getLogger()

    for i, orig_handler in enumerate(list(logger.handlers)):
        handler = MultiProcessingHandler(
            'mp-handler-{0}'.format(i), sub_handler=orig_handler)

        logger.removeHandler(orig_handler)
        logger.addHandler(handler)


class MultiProcessingHandler(logging.Handler):
    def __init__(self, name, sub_handler=None):
        logging.Handler.__init__(self)

        if sub_handler is None:
            sub_handler = logging.StreamHandler()
        self.sub_handler = sub_handler

        self.setLevel(self.sub_handler.level)
        self.setFormatter(self.sub_handler.formatter)

        self.queue = multiprocessing.Queue(-1)
        self._is_closed = False

        self._receive_thread = threading.Thread(target=self._receive)
        self._receive_thread.daemon = True
        self._receive_thread.start()

    def setFormatter(self, fmt):
        logging.Handler.setFormatter(self, fmt)
        self.sub_handler.setFormatter(fmt)

    def _receive(self):
        while not (self._is_closed and self.queue.empty()):
            try:
                record = self.queue.get(timeout=0.2)
                self.sub_handler.emit(record)
            except (KeyboardInterrupt, SystemExit):
                raise
            except EOFError:
                break
            except Queue.Empty:
                pass  # This periodically checks if the logger is closed.
            except Exception:
                traceback.print_exc(file=sys.stderr)

        self.queue.close()
        self.queue.join_thread()

    def send(self, s):
        self.queue.put_nowait(s)

    def _format_record(self, record):
        # ensure that exc_info and args have been stringified. Removes any
        # chance of unpickleable things inside and possibly reduces message
        # size sent over the pipe
        if record.args:
            record.msg = record.msg % record.args
            record.args = None
        if record.exc_info:
            self.format(record)
            record.exc_info = None

        return record

    def emit(self, record):
        try:
            s = self._format_record(record)
            self.send(s)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)

    def close(self):
        if not self._is_closed:
            self._is_closed = True
            self._receive_thread.join(5.0)  # Waits for receive queue to empty.

            self.sub_handler.close()
            logging.Handler.close(self)


def setup_mp_logging(logfile):
    '''
    Setup the logging and prepare it for use with multiprocessing
    '''

    # Setup the logging
    fmt = '%(asctime)s %(levelname)-8s %(threadName)12s %(funcName)15s(): ' \
        '%(message)s'
    formatter = logging.Formatter(fmt, datefmt='%m-%d %H:%M:%S')
    _logger.setLevel(logging.DEBUG)

    cHndlr = logging.StreamHandler()
    cHndlr.setLevel(logging.DEBUG)
    cHndlr.setFormatter(formatter)

    _logger.addHandler(cHndlr)

    fHndlr = logging.FileHandler(logfile, mode='w')
    fHndlr.setLevel(logging.DEBUG)
    fHndlr.setFormatter(formatter)

    _logger.addHandler(fHndlr)

    # Wrap the log handlers with the MPHandler, this is essential for the use
    # of multiprocessing, otherwise, tasks will hang.
    install_mp_handler(_logger)
