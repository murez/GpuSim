from .node import *

class Cluster:
    def __init__(self, total=-1, idle=-1, avail=-1):
        self.nodes = []
        self.total: int = total
        self.idle: int = idle
        # self.avail = avail

    def available(self, num_proc: int) -> bool:
        return self.idle >= num_proc

    def allocate(self, num_proc: int) -> bool:
        if not self.available(num_proc):
            return False
        self.idle -= num_proc
        return True

    def release(self, num_proc: int) -> bool:
        if(self.idle + num_proc > self.total):
            return False
        self.idle += num_proc
        return True
