"""
from progress_bar import ProgressBar

pbar = ProgressBar(steps)
pbar.upd()
"""

import os
import sys
import math
os.system('') #enable VT100 Escape Sequence for WINDOWS 10 Ver. 1607

from shutil import get_terminal_size
import time

import ipywidgets as ipy
import IPython
class ProgressIPy(object):
    def __init__(self, task_num=10):
        self.pbar = ipy.IntProgress(min=0, max=task_num, bar_style='') # (value=0, min=0, max=max, step=1, description=description, bar_style='')
        self.labl = ipy.Label()
        IPython.display.display(ipy.HBox([self.pbar, self.labl]))
        self.task_num = task_num
        self.completed = 0
        self.start()

    def start(self, task_num=None):
        if task_num is not None:
            self.task_num = task_num
        if self.task_num > 0:
            self.labl.value = '0/{}'.format(self.task_num)
        else:
            self.labl.value = 'completed: 0, elapsed: 0s'
        self.start_time = time.time()

    def upd(self, *p, **kw):
        self.completed += 1
        elapsed = time.time() - self.start_time + 0.0000000000001
        fps = self.completed / elapsed if elapsed>0 else 0
        if self.task_num > 0:
            finaltime = time.asctime(time.localtime(self.start_time + self.task_num * elapsed / float(self.completed)))
            fin = ' end %s' % finaltime[11:16]
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            self.labl.value = '{}/{}, rate {:.3g}s, time {}s, left {}s, {}'.format(self.completed, self.task_num, 1./fps, shortime(elapsed), shortime(eta), fin)
        else:
            self.labl.value = 'completed {}, time {}s, {:.1f} steps/s'.format(self.completed, int(elapsed + 0.5), fps)
        self.pbar.value += 1
        if self.completed == self.task_num: self.pbar.bar_style = 'success'
        return self.completed


class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''
    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal is small ({}), make it bigger for proper visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self, task_num=None):
        if task_num is not None:
            self.task_num = task_num
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def upd(self, msg=None):
        self.completed += 1
        elapsed = time.time() - self.start_time + 0.0000000000001
        fps = self.completed / elapsed if elapsed>0 else 0
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            finaltime = time.asctime(time.localtime(self.start_time + self.task_num * elapsed / float(self.completed)))
            fin_msg = ' %ss left, end %s' % (shortime(eta), finaltime[11:16])
            if msg is not None: fin_msg += '  ' + str(msg)
            mark_width = int(self.bar_width * percentage)
            bar_chars = 'X' * mark_width + '-' * (self.bar_width - mark_width) # ▒ ▓ █
            sys.stdout.write('\033[2A') # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            try:
                sys.stdout.write('[{}] {}/{}, rate {:.3g}s, time {}s, left {}s \n{}\n'.format(
                    bar_chars, self.completed, self.task_num, 1./fps, shortime(elapsed), shortime(eta), fin_msg))
            except:
                sys.stdout.write('[{}] {}/{}, rate {:.3g}s, time {}s, left {}s \n{}\n'.format(
                    bar_chars, self.completed, self.task_num, 1./fps, shortime(elapsed), shortime(eta), '<< unprintable >>'))
        else:
            sys.stdout.write('completed {}, time {}s, {:.1f} steps/s'.format(self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()

    def reset(self, count=None, newline=False):
        self.start_time = time.time()
        if count is not None:
            self.task_num = count
        if newline is True:
            sys.stdout.write('\n\n')

def time_days(sec):
    return '%dd %d:%02d:%02d' % (sec/86400, (sec/3600)%24, (sec/60)%60, sec%60)
def time_hrs(sec):
    return '%d:%02d:%02d' % (sec/3600, (sec/60)%60, sec%60)
def shortime(sec):
    if sec < 60:
        time_short = '%d' % (sec)
    elif sec < 3600:
        time_short  = '%d:%02d' % ((sec/60)%60, sec%60)
    elif sec < 86400:
        time_short  = time_hrs(sec)
    else:
        time_short = time_days(sec)
    return time_short

