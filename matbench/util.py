class RecursiveDotDict(dict):
    """
    Adapted from user Curt Hagenlocher from
    https://stackoverflow.com/questions/3031219/recursively-access-dict-via-attributes-as-well-as-index-access
    """
    MARKER = object()

    def __init__(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError('expected dict')

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, RecursiveDotDict):
            value = RecursiveDotDict(value)
        super(RecursiveDotDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        found = self.get(key, RecursiveDotDict.MARKER)
        if found is RecursiveDotDict.MARKER:
            found = RecursiveDotDict()
            super(RecursiveDotDict, self).__setitem__(key, found)
        return found

    __setattr__, __getattr__ = __setitem__, __getitem__
