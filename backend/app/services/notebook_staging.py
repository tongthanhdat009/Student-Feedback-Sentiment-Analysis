from pathlib import Path
from .notebook_inventory import NotebookInventory
class NotebookStaging:
    def stage(self, notebook_id: str) -> Path:
        return NotebookInventory().get_folder(notebook_id)
