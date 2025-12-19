
from pathlib import Path
import unittest

from pymmseqs.utils import run_mmseqs_command
from pymmseqs.commands import createdb
from vamb.taxvamb_easy import MMseqsRunner, Mmseqs
import os
import tempfile

PARENTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTDIR = Path(PARENTDIR) / "test" 

class TestDownloadDBMmseqs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_dir_resource = tempfile.TemporaryDirectory(prefix="mmseqs_test_")
        cls.tmpdir_runmmseqs = Path(cls._tmp_dir_resource.name) 
        cls.tmpdir_db = TESTDIR / "tmp" 
        cls.downloadlocation = TESTDIR / "database"
        cls.database = "Kalamari" # We use the kalmari db as it is small and takes ~1 min to download
        cls.filtered_db = TESTDIR / "database" / "filtered_db"
        if not cls.downloadlocation.exists():
            # Install the database
            Mmseqs().installDatabase(cls.downloadlocation, cls.tmpdir_db, cls.database)

            # Create a smaller version of it for querying. Here filtering only for eukaryotes
            MMseqsRunner().add_arguments([
                "filtertaxseqdb", cls.downloadlocation / cls.database, cls.filtered_db, "--taxon-list", "2759"
            ]).run()


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
        Mmseqs().removeTmpFiles(self.tmpdir_db, self.database)
        assert not self.tmpdir_db.exists()

    def test_run_mmseqs(self):
        # Given test dataset assign taxonomy
        contigs = TESTDIR / "data/mmseqs.fna"
        Mmseqs().assignTaxonomy(self.filtered_db, contigs, self.tmpdir_runmmseqs)



