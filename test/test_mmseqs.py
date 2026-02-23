
from pathlib import Path
from typing import override
import unittest

from pymmseqs.utils import run_mmseqs_command
from pymmseqs.commands import createdb
from vamb.mmseqs_wrapper import Database, MMseqsRunner, Mmseqs
import os
import tempfile

from vamb.taxonomy import Taxonomy
import vamb.vambtools
import vamb.parsecontigs

PARENTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTDIR = Path(PARENTDIR) / "test" 

class TestDownloadDBMmseqs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_dir_resource = tempfile.TemporaryDirectory(prefix="mmseqs_test_", dir="test", delete=False)
        cls.tmpdir_runmmseqs = Path(cls._tmp_dir_resource.name) 
        cls.tmpdir_db = TESTDIR / "tmp" 
        cls.downloadlocation = TESTDIR / "database"
        cls.database = Database.KALAMARI # We use the kalmari db as it is small and takes ~1 min to download
        cls.filtered_db = TESTDIR / "database" / "filtered_db"

        cls.mmseqs = Mmseqs(dbdir=cls.downloadlocation, database=cls.database)
        cls.filtered_mmseqs = Mmseqs(dbdir=cls.filtered_db.parent, database=Database.FILTERED_DB)

        if not cls.mmseqs.DatabaseExist() or \
            not cls.filtered_mmseqs.DatabaseExist():
            # Install the database
            cls.mmseqs.installDatabase(cls.downloadlocation, cls.tmpdir_db, cls.database)

            # Create a smaller version of it for querying. Here filtering only for eukaryotes
            MMseqsRunner().add_arguments([
                "filtertaxseqdb", cls.downloadlocation / cls.database.value, cls.filtered_db, "--taxon-list", "2759"
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
        self.mmseqs.removeTmpFiles(self.tmpdir_db, self.database)
        assert not self.tmpdir_db.exists()

    def test_run_mmseqs(self):
        # Given test dataset assign taxonomy
        contigs = TESTDIR / "data/mmseqs.fna"
        self.filtered_mmseqs.assignTaxonomy(contigs, self.tmpdir_runmmseqs)

class TestMmseqsTaxonomyReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.taxonomyfile = "test/data/mmseqs_lca.tsv"
        cls.fastafile = "test/data/mmseqs.fna"

    def test_load_mmseqs_taxonomy(self):
        parsed_tax = Taxonomy.parse_mmseqs_file(self.taxonomyfile, False)
        assert parsed_tax[1][1].ranks == ["unknown"]
        assert parsed_tax[0][0] == "KC153975.1"
        assert parsed_tax[0][1].ranks == [
        '-_cellular organisms',
         '-_Eukaryota',
         '-_Opisthokonta',
         'k_Metazoa',
         '-_Eumetazoa',
         '-_Bilateria',
         '-_Deuterostomia',
         'p_Chordata',
         '-_Craniata',
         '-_Vertebrata',
         '-_Gnathostomata',
         '-_Teleostomi',
         '-_Euteleostomi'
         ]
    def test_build_taxonomy(self):
        with vamb.vambtools.Reader(self.fastafile) as f:
          composition = vamb.parsecontigs.Composition.from_file(f, filename=None, minlength=20)
        tax = Taxonomy.from_file(self.taxonomyfile, composition.metadata, is_canonical=False, mmseqs_taxonomy=True)
        assert tax.contig_taxonomies[1].ranks == ["unknown"]
        assert tax.contig_taxonomies[0].ranks == [
        '-_cellular organisms',
         '-_Eukaryota',
         '-_Opisthokonta',
         'k_Metazoa',
         '-_Eumetazoa',
         '-_Bilateria',
         '-_Deuterostomia',
         'p_Chordata',
         '-_Craniata',
         '-_Vertebrata',
         '-_Gnathostomata',
         '-_Teleostomi',
         '-_Euteleostomi'
         ]

    def test_build_taxonomy_if_missmatch_in_contigs_due_to_length(self):
        with vamb.vambtools.Reader(self.fastafile) as f:
          composition = vamb.parsecontigs.Composition.from_file(f, filename=None, minlength=2000)
        tax = Taxonomy.from_file(self.taxonomyfile, composition.metadata, is_canonical=False, mmseqs_taxonomy=True)
        assert tax.contig_taxonomies[0].ranks == [
        '-_cellular organisms',
         '-_Eukaryota',
         '-_Opisthokonta',
         'k_Metazoa',
         '-_Eumetazoa',
         '-_Bilateria',
         '-_Deuterostomia',
         'p_Chordata',
         '-_Craniata',
         '-_Vertebrata',
         '-_Gnathostomata',
         '-_Teleostomi',
         '-_Euteleostomi'
         ]

