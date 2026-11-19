"""Migration failures must never look like a complete, verified asset import."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import import_assets
from check_assets import check_assets


class AssetMigrationTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root/'models').mkdir()
        (self.root/'source').mkdir()
        self.meshes = self.root/'source/meshes'
        self.meshes.mkdir()
        (self.meshes/'arm.stl').write_bytes(b'local-test-mesh')
        self.scene = self.root/'source/scene.xml'
        self.scene.write_text('<mujoco><compiler meshdir="meshes"/><asset>'
                              '<mesh name="arm" file="arm.stl"/></asset></mujoco>')
        self.destination = self.root/'models/cruzr_s2'

    def do_import(self):
        argv = ['import_assets', '--scene', str(self.scene), '--mesh-dir', str(self.meshes)]
        with patch.object(import_assets, 'ROOT', self.root), patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
            import_assets.main()

    def test_complete_import_and_corrupt_mesh_detection(self):
        self.do_import()
        result = check_assets(self.destination)
        self.assertEqual(result['meshes_verified'], 1)
        self.assertTrue(result['scene_hash_recorded_at_import'])
        (self.destination/'meshes_mjcf/arm.stl').write_bytes(b'corrupted in transfer')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            check_assets(self.destination)

    def test_source_copy_failure_leaves_no_partial_destination(self):
        with patch.object(import_assets.shutil, 'copy2', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.do_import()
        self.assertFalse(self.destination.exists())
        self.assertEqual(list((self.root/'models').iterdir()), [])
        self.assertTrue(self.scene.exists())

    def test_existing_assets_are_preserved(self):
        self.do_import()
        before = (self.destination/'scene.xml').read_bytes()
        with self.assertRaises(FileExistsError):
            self.do_import()
        self.assertEqual((self.destination/'scene.xml').read_bytes(), before)

    def test_scene_tamper_and_legacy_manifest_are_distinguished(self):
        self.do_import()
        scene = self.destination/'scene.xml'
        scene.write_bytes(scene.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError, 'scene checksum mismatch'):
            check_assets(self.destination)
        manifest_path = self.destination/'asset-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        del manifest['imported_scene_sha256']
        manifest_path.write_text(json.dumps(manifest))
        self.assertFalse(check_assets(self.destination)['scene_hash_recorded_at_import'])

    def test_absolute_paths_inside_package_are_not_portable(self):
        self.do_import()
        scene = self.destination/'scene.xml'
        text = scene.read_text()
        scene.write_text(text.replace('meshdir="meshes_mjcf"',
                                      f'meshdir="{self.destination / "meshes_mjcf"}"'))
        with self.assertRaisesRegex(ValueError, 'relative meshdir'):
            check_assets(self.destination)
        scene.write_text(text.replace('file="arm.stl"',
                                      f'file="{self.destination / "meshes_mjcf/arm.stl"}"'))
        manifest_path = self.destination/'asset-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['meshes'][0]['file'] = str(self.destination/'meshes_mjcf/arm.stl')
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'absolute mesh reference'):
            check_assets(self.destination)

    def test_unpackaged_external_dependencies_are_rejected_before_copying(self):
        text = self.scene.read_text()
        for dependency in ('<texture name="floor" type="2d" file="floor.png"/>',
                           '<texture name="sky" type="cube" fileback="sky.png"/>',
                           '<hfield name="terrain" file="terrain.png"/>',
                           '<include file="robot.xml"/>'):
            self.scene.write_text(text.replace('</asset>', dependency+'</asset>'))
            with self.subTest(dependency=dependency), self.assertRaises(ValueError):
                self.do_import()
            self.assertFalse(self.destination.exists())

    def test_empty_recorded_scene_hash_cannot_disable_verification(self):
        self.do_import()
        path = self.destination/'asset-manifest.json'
        manifest = json.loads(path.read_text())
        manifest['imported_scene_sha256'] = ''
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'scene checksum mismatch'):
            check_assets(self.destination)
