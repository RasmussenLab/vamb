
from pathlib import Path
import unittest

from pymmseqs.utils import run_mmseqs_command
from vamb.taxvamb_easy import MMseqsRunner, Mmseqs
import os

PARENTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTDIR = Path(PARENTDIR) / "test" 

class TestMmseqs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = TESTDIR / "tmp"
        cls.downloadlocation = TESTDIR / "database"
        cls.database = "Kalamari" # We use the kalmari db as it is small and takes ~1 min to download
        if not cls.downloadlocation.exists():
            Mmseqs().installDatabase(cls.downloadlocation, cls.tmpdir, cls.database)
        
    def test_download_db(self):
        files_which_should_exist = [
            "Kalamari",
            "Kalamari.dbtype",
            "Kalamari.index",
            "Kalamari.lookup",
            "Kalamari.source",
            "Kalamari.version",
            "Kalamari_h",
            "Kalamari_h.dbtype",
            "Kalamari_h.index",
            "Kalamari_mapping",
            "Kalamari_taxonomy",
        ]
        for file in files_which_should_exist:
            assert (self.downloadlocation / file).exists()

    def test_remove_tmp_files(self):
        Mmseqs().removeTmpFiles(self.tmpdir, self.database)
        assert not self.tmpdir.exists()








    # def test_mmseqs_works(self):
    #     result = run_mmseqs_command(["--help"])
    #     print(f"{result.stdout!r}")
