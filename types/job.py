from enum import Enum

__metaclass__ = type

class JobState(Enum):
    SUBMITTED = 1
    RUNNING = 2
    FINISHED = 3

class Job:
    def __init__(self, id, submit_time, wait, run, used_proc, used_ave_cpu, used_mem, req_proc, req_time, req_mem, status, user_id, group_id, num_exe, num_queue, num_part, num_pre, think_time, start_time=-1, end_time=-1, score=0, state=0, happy=-1, est_start=-1):
        self.id: int                = id
        self.submit_time: float     = submit_time
        self.wait: float            = wait
        self.run: float             = run
        self.used_proc: int         = used_proc
        self.used_ave_cpu: float    = used_ave_cpu
        self.used_mem: float        = used_mem
        self.req_proc: int          = req_proc
        self.req_time: float        = req_time
        self.req_mem: float         = req_mem
        self.status: int            = status
        self.user_id: int           = user_id
        self.group_id: int          = group_id
        self.num_exe: int           = num_exe
        self.num_queue: int         = num_queue
        self.num_part: int          = num_part
        self.num_pre: int           = num_pre
        self.think_time: int        = think_time
        self.start_time: int        = start_time
        self.end_time: int          = end_time
        self.score: int             = score
        self.state: int | JobState  = state
        self.happy: int             = happy
        self.est_start: int         = est_start

    def submit(self, score=0, est_start=-1):
        self.state = JobState.SUBMITTED
        self.score = score
        self.est_start = est_start
        return 0

    def start(self, time):
        self.state = JobState.RUNNING
        self.start_time = time
        self.wait = time - self.submit_time
        self.end = time + self.run
        return 0

    def finish(self, time=None):
        self.state = JobState.FINISHED
        if time:
            self.end = time
        return 0