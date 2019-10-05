def vprint(*args, verbose=False):
    '''
    verbose = True
    args = 2,"abc","{}".format(2)2 abc 2
    vprint(*args,verbose=True)
    print(*args)
    print(2,"abc","{}".format(2))
    vprint(*args,verbose=0)
    '''
    # TODO fix bug when verbose is arg,not kwarg
    # eg. vprint(*args,verbose)
    if verbose:
        print(*args[:])


import time


# %load -r 78-111 /home/btd26/xai-research/xai/xai/utils/utils.py
class Timer:
    '''
    # Start timer
  my_timer = Timer()

  # ... do something

  # Get time string:
  time_hhmmss = my_timer.get_time_hhmmss()
  print("Time elapsed: %s" % time_hhmmss )

  # ... use the timer again
  my_timer.restart()

  # ... do something

  # Get time:
  time_hhmmss = my_timer.get_time_hhmmss()

  # ... etc
    '''

    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def get_time_hhmmss(self):
        end = time.time()
        m, s = divmod(end - self.start, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        return time_str


# TODO extract in config.inin
from pathlib import Path
import os
FIG_FOLDER = path_dir = Path(os.path.dirname(os.path.abspath("__file__"))).parents[0] / "dginn/figures/"

def make_dir(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)