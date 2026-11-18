"""Review-corrected public entrypoint. See README before running."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('jdft2d_review_reproduce.py')), run_name='__main__')
