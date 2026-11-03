"""Keep the SAM 3 URL, gitlink and documented runtime revision consistent."""

import configparser
from pathlib import Path
import shutil
import subprocess
import unittest

from tools.sam3_runtime import SAM3_REVISION


ROOT = Path(__file__).resolve().parents[1]


class SubmoduleSetupTests(unittest.TestCase):
    def test_sam3_has_a_restorable_official_submodule_mapping(self):
        config = configparser.ConfigParser()
        with (ROOT / ".gitmodules").open(encoding="utf-8") as source:
            config.read_file(source)
        section = config['submodule "sam3"']
        self.assertEqual(section["path"], "sam3")
        self.assertEqual(section["url"], "https://github.com/facebookresearch/sam3.git")

    @unittest.skipUnless(shutil.which("git"), "Git is required to inspect the submodule gitlink")
    def test_gitlink_matches_the_runtime_revision(self):
        if not (ROOT / ".git").exists():
            self.skipTest("Source archive has no Git index")
        result = subprocess.run(
            ["git", "ls-files", "--stage", "--", "sam3"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        records = result.stdout.strip().splitlines()
        self.assertEqual(len(records), 1, "SAM 3 must remain one pinned submodule")
        metadata, path = records[0].split("\t", 1)
        mode, revision, stage = metadata.split()
        self.assertEqual((mode, stage, path), ("160000", "0", "sam3"))
        self.assertEqual(revision, SAM3_REVISION)


if __name__ == "__main__":
    unittest.main()
