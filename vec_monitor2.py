import time
from collections import deque, namedtuple

import numpy as np
import gym3

Episode = namedtuple("Episode", ["ret", "len", "time", "info"])


class PostActProcessing(gym3.Wrapper):
    """
    Call process() after each action, except possibly possibly the last 
    one which you never called observe for.
    """

    def __init__(self, env):
        super().__init__(env)
        self.need_process = False

    def process_if_needed(self):
        if self.need_process:
            self.process()
            self.need_process = False

    def act(self, ac):
        self.process_if_needed()
        self.env.act(ac)
        self.need_process = True

    def observe(self):
        self.process_if_needed()
        return self.env.observe()

    def process(self):
        raise NotImplementedError


class VecMonitor2(PostActProcessing):
    def __init__(self, venv, keep_buf=0):
        """
        use n_per_env if you want to keep sep
        """
        super().__init__(venv)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()
        if keep_buf:
            self.ep_buf = deque([], maxlen=keep_buf)
        else:
            self.ep_buf = None

        self.eprets = np.zeros(self.num, "f")
        self.eplens = np.zeros(self.num, "i")

    def process(self):
        lastrews, _obs, firsts = self.env.observe()
        infos = self.env.get_info()
        self.eprets += lastrews
        self.eplens += 1
        for i in range(self.num):
            if firsts[i]:
                timefromstart = round(time.time() - self.tstart, 6)
                ep = Episode(self.eprets[i], self.eplens[i], timefromstart, infos[i])
                if self.ep_buf is not None:
                    self.ep_buf.append(ep)
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0

    def clear_episode_bufs(self):
        if self.ep_buf:
            self.ep_buf.clear()
        self.clear_per_env_episode_buf()

    def clear_per_env_episode_buf(self):
        if self.per_env_buf:
            for i in range(self.num):
                self.per_env_buf[i].clear()

