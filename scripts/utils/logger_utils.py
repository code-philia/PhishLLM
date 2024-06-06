import logging
import re

class TxtColors:
	OK = '\033[92m'
	DEBUG = '\033[94m'
	WARNING = "\033[93m"
	FATAL = '\033[91m'
	EXCEPTION = '\033[100m'
	ENDC = '\033[0m'

'''Logging Utils'''
class PhishLLMLogger():
    _caller_prefix = "PhishLLMLogger"
    _verbose = True
    _logfile = None
    _debug = False # Off by default
    _warning = True

    @classmethod
    def set_verbose(cls, verbose):
        cls._verbose = verbose

    @classmethod
    def set_logfile(cls, logfile):
        # if os.path.isfile(logfile):
        #     os.remove(logfile)  # Remove the existing log file
        PhishLLMLogger._logfile = logfile

    @classmethod
    def unset_logfile(cls):
        PhishLLMLogger.set_logfile(None)

    @classmethod
    def set_debug_on(cls):
        PhishLLMLogger._debug = True

    @classmethod
    def set_debug_off(cls): # Call if need to turn debug messages off
        PhishLLMLogger._debug = False

    @classmethod
    def set_warning_on(cls):
        PhishLLMLogger._warning = True

    @classmethod
    def set_warning_off(cls): # Call if need to turn warnings off
        PhishLLMLogger._warning = False

    @classmethod
    def spit(cls, msg, warning=False, debug=False, error=False, exception=False, caller_prefix=""):
        logging.basicConfig(level=logging.DEBUG if PhishLLMLogger._debug else logging.WARNING)
        caller_prefix = f"[{caller_prefix}]" if caller_prefix else ""
        prefix = "[FATAL]" if error else "[DEBUG]" if debug else "[WARNING]" if warning else "[EXCEPTION]" if exception else ""
        logger = logging.getLogger("custom_logger")  # Choose an appropriate logger name
        if PhishLLMLogger._logfile:
            log_msg = re.sub(r"\033\[\d+m", "", msg)
            log_handler = logging.FileHandler(PhishLLMLogger._logfile, mode='a')
            log_formatter = logging.Formatter('%(message)s')
            log_handler.setFormatter(log_formatter)
            logger.addHandler(log_handler)
            logger.propagate = False
            logger.setLevel(logging.DEBUG if PhishLLMLogger._debug else logging.WARNING)
            logger.debug("%s%s %s" % (caller_prefix, prefix, log_msg))
            logger.removeHandler(log_handler)
        else:
            if PhishLLMLogger._verbose:
                txtcolor = TxtColors.FATAL if error else TxtColors.DEBUG if debug else TxtColors.WARNING if warning else "[EXCEPTION]" if exception else TxtColors.OK
                # if not debug or Logger._debug:
                if (not debug and not warning) or (debug and PhishLLMLogger._debug) or (warning and PhishLLMLogger._warning):
                    print("%s%s%s %s" % (txtcolor, caller_prefix, prefix, msg))