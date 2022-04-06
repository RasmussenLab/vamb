import sys
import os
import unittest

PARENTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENTDIR)

DATADIR = os.path.join(PARENTDIR, "test", "data")

if __name__ == "__main__":
    unittest.main(module='vambtools', exit=False)
    unittest.main(module='parsecontigs', exit=False)
    unittest.main(module='parsebam', exit=False)
    unittest.main(module='encode', exit=False)
    unittest.main(module='cluster', exit=False)
