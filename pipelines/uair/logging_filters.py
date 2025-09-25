import logging


class ModuloFilter(logging.Filter):
    def __init__(self, name: str = "", mod: int = 10):
        super().__init__(name)
        self.mod = max(1, int(mod))
        self._count = 0

    def filter(self, record: logging.LogRecord) -> bool:
        # Always allow warnings/errors
        if record.levelno >= logging.WARNING:
            return True
        self._count += 1
        return (self._count % self.mod) == 0

class PatternModuloFilter(logging.Filter):
    def __init__(self, name: str = "", mod: int = 10, pattern: str = "Elapsed time for batch"):
        super().__init__(name)
        self.mod = max(1, int(mod))
        self.pattern = str(pattern)
        self._count = 0

    def filter(self, record: logging.LogRecord) -> bool:
        # Always allow warnings/errors
        if record.levelno >= logging.WARNING:
            return True
        try:
            msg = str(record.getMessage())
        except Exception:
            msg = str(record.msg)
        if self.pattern and (self.pattern in msg):
            self._count += 1
            return (self._count % self.mod) == 0
        return True


