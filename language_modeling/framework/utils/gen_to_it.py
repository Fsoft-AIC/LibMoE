class GenToIt:
    def __init__(self, gen, *args, **kwargs):
        self.gen = gen
        self.args  = args
        self.kwargs = kwargs
        self.gen_inst = None
        self.__iter__()
        self.initialized = False

    def __iter__(self):
        assert (self.gen_inst is None) or (self.initialized == False)
        self.initialized = True
        self.gen_inst = self.gen(*self.args, **self.kwargs)
        return self

    def __next__(self):
        try:
            n = next(self.gen_inst)
            return n
        except StopIteration:
            self.gen_inst = None
            raise
