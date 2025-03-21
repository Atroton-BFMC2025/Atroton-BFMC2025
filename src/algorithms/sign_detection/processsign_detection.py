if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../../..")

from src.templates.workerprocess import WorkerProcess
from src.algorithms.sign_detection.threads.threadsign_detection import threadsign_detection

class processsign_detection(WorkerProcess):
    """This process handles sign_detection.
    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
        logging (logging object): Made for debugging.
        debugging (bool, optional): A flag for debugging. Defaults to False.
    """

    def __init__(self, queueList, logging, debugging=False):
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        super(processsign_detection, self).__init__(self.queuesList)

    def run(self):
        """Apply the initializing methods and start the threads."""
        super(processsign_detection, self).run()

    def _init_threads(self):
        """Create the sign_detection Publisher thread and add to the list of threads."""
        sign_detectionTh = threadsign_detection(
            self.queuesList, self.logging, self.debugging
        )
        self.threads.append(sign_detectionTh)
