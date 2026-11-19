#!/usr/bin/env python3
"""Import authorized local model assets atomically; preserve the source files."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

from check_assets import check_assets, mesh_references, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--source-root', type=Path, help='Directory containing show-harness and cruzr_urdf')
    group.add_argument('--scene', type=Path, help='Alternative local standalone scene.xml')
    parser.add_argument('--mesh-dir', type=Path, help='Mesh source directory; required with --scene')
    args = parser.parse_args()
    if args.source_root:
        if args.mesh_dir:
            parser.error('--mesh-dir is only used with --scene')
        source = args.source_root.resolve()
        scene = source / 'show-harness/sim/cruzr_s2_urdf.xml'
        meshdir = source / 'cruzr_urdf/cruzr_s2_description_gripper/meshes_mjcf'
    else:
        if not args.mesh_dir:
            parser.error('--scene requires --mesh-dir')
        scene, meshdir = args.scene.resolve(), args.mesh_dir.resolve()
    destination = ROOT / 'models/cruzr_s2'
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite existing assets: {destination}')
    tree = ET.parse(scene)
    compiler = tree.find('compiler')
    if compiler is None:
        raise ValueError('Scene has no compiler element')
    meshes = sorted(mesh_references(tree))
    if not meshes:
        raise ValueError('Scene has no mesh assets')
    for name in meshes:
        if Path(name).name != name or not (meshdir / name).is_file():
            raise FileNotFoundError(f'Missing or invalid mesh: {name}')
    # Validate before copying and remove all staging files on failure. Never
    # leave a partially imported destination that appears ready to simulate.
    with tempfile.TemporaryDirectory(prefix='.asset-import-', dir=destination.parent) as temporary:
        staging = Path(temporary) / 'cruzr_s2'
        (staging / 'meshes_mjcf').mkdir(parents=True)
        records = []
        for name in meshes:
            copied = staging / 'meshes_mjcf' / name
            shutil.copy2(meshdir / name, copied)
            records.append({'file': name, 'sha256': sha256(copied)})
        compiler.set('meshdir', 'meshes_mjcf')
        tree.write(staging / 'scene.xml', encoding='utf-8', xml_declaration=True)
        (staging / 'asset-manifest.json').write_text(json.dumps({
            'source_scene': str(scene), 'source_scene_sha256': sha256(scene),
            'imported_scene_sha256': sha256(staging / 'scene.xml'),
            'license': 'Unverified; local reuse only, not relicensed by this repository',
            'meshes': records,
        }, indent=2)+'\n', encoding='utf-8')
        check_assets(staging)
        staging.rename(destination)
    print(f'Imported {len(meshes)} meshes and scene to {destination}')


if __name__ == '__main__':
    main()
