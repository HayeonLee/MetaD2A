from pathlib import Path
import sys
dir_path = (Path(__file__).parent).resolve()
if str(dir_path) not in sys.path: sys.path.insert(0, str(dir_path))

from .architecture import train_single_model