class Node:
    def __init__(self, id, location, group, state, proc, start=-1, end=-1, extend=None):
        self.id       = id
        self.location = location
        self.group    = group
        self.state    = state
        self.proc     = proc
        self.start    = start
        self.end      = end
        self.extend   = extend