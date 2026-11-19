#!/usr/bin/env python3
"""Verify local robot asset hashes without loading MuJoCo or touching sources."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def mesh_references(tree):
    """Only standalone mesh-based MJCF packages are supported by the importer."""
    if tree.findall('.//include'):
        raise ValueError('Asset package must be standalone; expand include files first')
    # These dependencies are not copied or hashed by the current importer.
    for tag in ('texture', 'hfield', 'skin', 'model'):
        if any(value for node in tree.findall(f'.//asset/{tag}')
               for key, value in node.attrib.items() if key.startswith('file')):
            raise ValueError(f'Unsupported external {tag} file; package it explicitly before importing')
    return {m.attrib['file'] for m in tree.findall('.//asset/mesh') if m.get('file')}


def check_assets(directory, *, scene_name='scene.xml'):
    directory = Path(directory).resolve()
    if Path(scene_name).name != scene_name:
        raise ValueError('scene_name must be a filename inside the asset directory')
    scene = directory / scene_name
    manifest = json.loads((directory / 'asset-manifest.json').read_text(encoding='utf-8'))
    tree = ET.parse(scene)
    compiler = tree.find('compiler')
    if compiler is None:
        raise ValueError('Scene has no compiler element')
    relative_meshdir = Path(compiler.get('meshdir', '.'))
    if relative_meshdir.is_absolute():
        raise ValueError('Asset package requires a relative meshdir for migration')
    meshdir = (directory / relative_meshdir).resolve()
    if not meshdir.is_relative_to(directory):
        raise ValueError('Asset package uses nonportable meshdir outside its directory')
    records = {record['file']: record['sha256'] for record in manifest['meshes']}
    if len(records) != len(manifest['meshes']):
        raise ValueError('Duplicate mesh records in manifest')
    files = mesh_references(tree)
    if not files or set(records) != files:
        raise ValueError('Manifest does not match scene mesh references')
    for name in sorted(files):
        if Path(name).is_absolute():
            raise ValueError(f'Nonportable absolute mesh reference: {name}')
        path = (meshdir / name).resolve()
        if not path.is_relative_to(meshdir) or not path.is_file():
            raise ValueError(f'Missing or nonportable mesh: {name}')
        if sha256(path) != records[name]:
            raise ValueError(f'Mesh checksum mismatch: {name}')
    scene_hash = sha256(scene)
    if 'imported_scene_sha256' in manifest and scene_hash != manifest['imported_scene_sha256']:
        raise ValueError('Imported scene checksum mismatch')
    return {'success': True, 'meshes_verified': len(files), 'scene_sha256': scene_hash,
            'scene_hash_recorded_at_import': 'imported_scene_sha256' in manifest,
            'manifest_sha256': sha256(directory / 'asset-manifest.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-dir', type=Path, default=ROOT / 'models/cruzr_s2')
    parser.add_argument('--scene-name', default='scene.xml')
    args = parser.parse_args()
    try:
        result = check_assets(args.asset_dir, scene_name=args.scene_name)
    except Exception as exc:
        result = {'success': False, 'error': f'{type(exc).__name__}: {exc}'}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['success'] else 1)
