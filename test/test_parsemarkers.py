import unittest
import vamb
import testtools
from pathlib import Path
import tempfile
import shutil
import io


class TestParseMarkers(unittest.TestCase):
    def test_instantiate(self):
        tmp = tempfile.mkdtemp()
        tmp_path = Path(tmp)
        shutil.rmtree(tmp)
        markers = vamb.parsemarkers.Markers.from_files(
            Path(testtools.DATADIR).joinpath("marker.fna"),
            Path(testtools.PARENTDIR).joinpath("vamb").joinpath("marker.hmm"),
            ["abc"],
            tmp_path,
            4,
            None,
        )
        self.assertIsNotNone(markers.markers[0])
        self.assertEqual(len(markers.markers), 1)
        self.assertEqual(set(markers.markers[0]), {39})
        self.assertEqual(
            markers.refhash, vamb.vambtools.RefHasher.hash_refnames(["abc"])
        )

        buf = io.StringIO()
        markers.save(buf)
        buf.seek(0)

        markers2 = vamb.parsemarkers.Markers.load(buf, markers.refhash)
        self.assertEqual(len(markers.markers), len(markers2.markers))
        self.assertEqual(set(markers.markers[0]), set(markers2.markers[0]))
        self.assertEqual(markers.marker_names, markers2.marker_names)
