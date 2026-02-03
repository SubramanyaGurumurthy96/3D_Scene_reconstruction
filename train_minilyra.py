import os
import sys

REPO_ROOT = os.path.dirname(__file__)
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from scene_reconstruction.train_minilyra import main


if __name__ == "__main__":
    main()
