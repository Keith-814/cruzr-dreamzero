#!/usr/bin/env python3
"""Copy existing local model assets without distributing them through Git."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Existing directory containing show-harness and cruzr_urdf")
    args = parser.parse_args()
    source = args.source_root.resolve()
    destination = Path(__file__).resolve().parents[1] / "models" / "cruzr_s2"
    scene = source / "show-harness/sim/cruzr_s2_urdf.xml"
    meshdir = source / "cruzr_urdf/cruzr_s2_description_gripper/meshes_mjcf"
    tree = ET.parse(scene)
    meshes = sorted({m.attrib["file"] for m in tree.findall(".//asset/mesh")})
    for name in meshes:
        if Path(name).name != name or not (meshdir / name).is_file():
            raise FileNotFoundError(f"Missing or invalid mesh: {name}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing assets: {destination}")
    (destination / "meshes_mjcf").mkdir(parents=True, exist_ok=True)
    records = []
    for name in meshes:
        src = meshdir / name
        shutil.copy2(src, destination / "meshes_mjcf" / name)
        records.append({"file": name, "sha256": hashlib.sha256(src.read_bytes()).hexdigest()})
    compiler = tree.find("compiler")
    if compiler is None:
        raise ValueError("Scene has no compiler configuration")
    compiler.set("meshdir", "meshes_mjcf")
    tree.write(destination / "scene.xml", encoding="utf-8", xml_declaration=True)
    (destination / "asset-manifest.json").write_text(json.dumps({
        "source_scene": str(scene), "source_scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "license": "Unverified; local reuse only, not relicensed by this repository", "meshes": records
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Imported {len(meshes)} meshes and scene to {destination}")


if __name__ == "__main__":
    main()
