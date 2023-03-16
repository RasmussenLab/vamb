import vamb
import re
import subprocess
import unittest


def grep(path, regex):
    with open(path) as file:
        for line in file:
            m = regex.search(line)
            if m is not None:
                g = m.groups()
                return (int(g[0]), int(g[1]), int(g[2]))

    raise ValueError(f"Could not find regex in path {path}")


def snakemake_vamb_version(path):
    regex = re.compile(
        r"https://github\.com/RasmussenLab/vamb/archive/v([0-9]+)\.([0-9]+)\.([0-9]+)\.zip"
    )
    return grep(path, regex)


def changelog_version(path):
    with open(path) as file:
        next(file)  # header
        textline = next(filter(None, map(str.strip, file)))
    regex = re.compile(r"v([0-9]+)\.([0-9]+)\.([0-9]+)*(?:-([A-Za-z]+))")
    m = regex.search(textline)
    if m is None:
        raise ValueError("Could not find version in first non-header line of CHANGELOG")
    g = m.groups()
    v_nums = (int(g[0]), int(g[1]), int(g[2]))
    return v_nums if g[3] is None else (*v_nums, g[3])


def readme_vamb_version(path):
    regex = re.compile(
        r"https://github\.com/RasmussenLab/vamb/archive/v([0-9]+)\.([0-9]+)\.([0-9]+)\.zip"
    )
    return grep(path, regex)


def validate_init(init):
    if not (
        isinstance(init, tuple)
        and len(init) in (3, 4)
        and all(isinstance(i, int) for i in init[:3])
    ):
        raise ValueError("Wrong format of __version__ in __init__.py")


def latest_git_tag():
    st = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"], capture_output=True
    ).stdout.decode()
    regex = re.compile(r"^v?([0-9]+)\.([0-9]+)\.([0-9])\n?$")
    m = regex.match(st)
    if m is None:
        raise ValueError("Could not find last git tag")
    else:
        return tuple(int(i) for i in m.groups())


def head_git_tag():
    st = subprocess.run(
        ["git", "tag", "--points-at", "HEAD"], capture_output=True
    ).stdout.decode()
    if len(st) == 0:
        return (None, None)
    regex = re.compile(r"^(v([0-9]+)\.([0-9]+)\.([0-9]))\n?$")
    m = regex.match(st)
    if m is None:
        raise ValueError("HEADs git tag is not a valid version number")
    return tuple(int(i) for i in m.groups()[1:4])


class TestVersions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        validate_init(vamb.__version__)
        cls.v_init = vamb.__version__
        cls.v_changelog = changelog_version("../CHANGELOG.md")
        cls.last_tag = latest_git_tag()
        head_tag = head_git_tag()
        cls.head_tag = head_tag

    def test_same_versions(self):
        # The version in the changelog must fit the one in __init__
        self.assertEqual(self.v_init, self.v_changelog)

    def test_dev_version(self):
        # If the current version is a DEV version, it must be a greater version
        # than the latest release.
        # If not, it must be the same version as the tag of the current commit,
        # i.e. the current commit must be a release version.
        if len(self.v_init) > 3:
            self.assertGreater(self.v_init[:3], self.last_tag)
        else:
            self.assertEqual(self.v_init, self.head_tag)
            self.assertEqual(self.v_init[:3], self.last_tag)
