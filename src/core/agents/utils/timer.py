import time

class Timer:
    def __init__(self, enable_print: bool = True):
        self.start_time = None
        self.end_time = None
        self.note = None
        self.enable_print = enable_print

    def start(self, note: str = ""):
        """Start timing with an optional note."""
        self.start_time = time.time()
        self.end_time = None
        self.note = note
        if self.enable_print:
            if note:
                print(f"Timer started... (note: {note})")
            else:
                print("Timer started...")

    def end(self):
        """Stop timing and return the duration in seconds."""
        if self.start_time is None:
            raise ValueError("Please call start() before calling end()")
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        if self.enable_print:
            if self.note:
                print(f"Timer ended (note: {self.note}), duration: {duration:.4f} seconds")
            else:
                print(f"Timer ended, duration: {duration:.4f} seconds")
        return duration
