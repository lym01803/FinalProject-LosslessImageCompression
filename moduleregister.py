class Register(object):
    record = dict()
    def __init__(self):
        super().__init__()

    @classmethod
    def register(cls, obj):
        try:
            cls.record[obj.__name__] = obj
            return obj
        except Exception as e:
            print(e)
            pass
    
    @classmethod
    def get(cls, key):
        obj = cls.record.get(key, None)
        if obj:
            return obj
        else:
            raise Exception(f'Can not find object {key}')

