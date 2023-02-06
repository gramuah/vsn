# Send output to file
class Unbuffered:
    def __init__(self, stream, logger):
        self.stream = stream
        self.logger = logger

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.logger.write(data)  # Write the data of stdout here to a text file as well

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
