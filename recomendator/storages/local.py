class Local():
    """
    Local storage.
    """
    def __init__(self):
        """

        """
        self.storage = {}

    def get(self, key):
        """
        Get by key.
        """
        return self.storage.get(key, {})

    def post(self, key, value):
        """
        Post value by key.
        """
        self.storage[key] = value

    def delete(self):
        """
        Delete by key.
        """
        del(self.storage[key])

    def store(self, sDict):
        self.storage = sDict

    def generator(self):
        """
        Local storage generator.
        """
        for key in self.storage.keys():
            yield key