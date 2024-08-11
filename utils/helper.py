from contextlib import contextmanager
import sys, os
import warnings



####################################################################### Functions #######################################################################


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:  
            sys.stdout = devnull
            sys.stderr = devnull        
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


####################################################################### Functions #######################################################################


@contextmanager
def suppress_std(target = ["out"]): 
    with open(os.devnull, "w") as devnull:
        
        if "out" in target: old_stdout = sys.stdout
        if "err" in target: old_stderr = sys.stderr
        
        try:  
            if "out" in target: sys.stdout = devnull
            if "err" in target: sys.stdout = devnull        
            yield
            
        finally:
            if "out" in target: sys.stdout = old_stdout
            if "err" in target: sys.stderr = old_stderr


####################################################################### Functions #######################################################################


def track(lim, string = 'Iteration'):
    print(f'{string}: ', end = '')
    for i in range(1, lim+1):
        ''' Any operation(s) requiring a loop '''
        # print(f"\r Iteration : [{'='*i}>{' '*(100-i)}] {i}/100", end = '')
        print('\r{}'.format(string), end = '')


####################################################################### Functions #######################################################################


# Suppress all warnings globally
warnings.filterwarnings("ignore")

@contextmanager
def force_suppress():
    """Suppress all output by redirecting stdout and stderr to os.devnull."""
    with open(os.devnull, 'w') as fnull:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(fnull.fileno(), 1)
        os.dup2(fnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)
