
class AppState:
    def __init__(self):
        self.chunks: list = []
        self.embeddings = None  # numpy array
        self.file_context: str = ""
        self.memory: list = []



