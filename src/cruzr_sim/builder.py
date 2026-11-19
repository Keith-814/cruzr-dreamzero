"""Generate a tabletop scene around an imported robot-only MJCF asset."""
import xml.etree.ElementTree as ET
from pathlib import Path
from .scene import ARM_JOINTS


def values(items):
    return " ".join(str(x) for x in items)


def build_xml(cfg):
    source = Path(cfg["simulation"]["model"])
    if not source.is_file():
        raise FileNotFoundError(f"{source}: run scripts/import_assets.py --source-root /data1/oylx2")
    root = ET.parse(source).getroot()
    root.set("model", "cruzr_tabletop")
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("Imported MJCF must contain a compiler element and relative meshdir")
    meshdir = Path(compiler.get("meshdir", "."))
    compiler.set("meshdir", str((source.parent / meshdir).resolve()))
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(cfg["simulation"]["physics_dt"]))
    option.set("integrator", "implicitfast")
    option.set("gravity", "0 0 -9.81")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    glob = visual.find("global")
    if glob is None:
        glob = ET.SubElement(visual, "global")
    camera = cfg["camera"]
    # MuJoCo's offscreen buffer must be large enough for the largest stream.
    glob.set("offwidth", str(max(camera["width"], camera["head_width"], camera["wrist_width"])))
    glob.set("offheight", str(max(camera["height"], camera["head_height"], camera["wrist_height"])))
    world = root.find("worldbody")
    if world is None:
        raise ValueError("Imported MJCF has no worldbody")
    robot = world.find("body[@name='base_link']")
    if robot is None:
        raise ValueError("Imported robot has no base_link")
    for child in list(world):
        if child is not robot:
            world.remove(child)
    robot.set("pos", values(cfg["robot"]["base_position"]))
    robot.set("quat", values(cfg["robot"]["base_quaternion_wxyz"]))
    # Apply SDK bounds by joint name; preserve narrower asset bounds.
    joints = {j.get("name"): j for j in root.findall(".//joint")}
    actuators = {a.get("joint"): a for a in root.find("actuator")}
    for i, name in enumerate(ARM_JOINTS):
        key = "R_" + name + "_joint"
        j, a = joints[key], actuators[key]
        old_lo, old_hi = map(float, j.get("range").split())
        lo = max(old_lo, cfg["robot"]["lower"][i])
        hi = min(old_hi, cfg["robot"]["upper"][i])
        if not lo <= cfg["robot"]["home"][i] <= hi:
            raise ValueError(f"home outside SDK/model bounds: {key}")
        j.set("range", values([lo, hi]))
        a.set("kp", str(cfg["robot"]["kp"][i]))
        a.set("kv", str(cfg["robot"]["kv"][i]))
        cap = cfg["robot"]["force_limit"][i]
        a.set("forcerange", values([-cap, cap]))
        a.set("ctrlrange", values([lo, hi]))
        a.set("ctrllimited", "true")
    g = cfg["gripper"]
    for side in ("L", "R"):
        for index in (1, 2):
            key = f"{side}_finger{index}_joint"
            low, high = (g["zero_gap"] - g["max_width"]) / 2, g["zero_gap"] / 2
            joints[key].set("range", values([low, high]))
            a = actuators[key]
            for name in ("kp", "kv"):
                a.set(name, str(g[name]))
            a.set("forcerange", values([-g["force_limit"], g["force_limit"]]))
            a.set("ctrlrange", values([low, high]))
            a.set("ctrllimited", "true")
    task, camera = cfg["task"], cfg["camera"]
    ET.SubElement(world, "light", pos="0.5 0.3 2.4", dir="-0.3 0.1 -1", directional="true")
    ET.SubElement(world, "geom", name="table", type="box", pos=values(task["table_position"]),
                  size=values(task["table_half_size"]), rgba="0.7 0.7 0.72 1", contype="2", conaffinity="3")
    cube = ET.SubElement(world, "body", name="cube", pos=values(task["cube_position"]))
    ET.SubElement(cube, "freejoint", name="cube_free")
    ET.SubElement(cube, "geom", name="cube_geom", type="box", size=values([task["cube_half_size"]]*3),
                  mass=str(task["cube_mass"]), rgba="0.95 0.45 0.1 1", contype="2", conaffinity="3",
                  friction=values(task["friction"]))
    plate = ET.SubElement(world, "body", name="plate", pos=values(task["target_position"]))
    ET.SubElement(plate, "geom", type="cylinder", size=values([task["target_radius"], 0.008]),
                  rgba="0.05 0.28 0.95 1", contype="0", conaffinity="0")
    ET.SubElement(world, "camera", name="front_cam", pos=values(camera["front_position"]),
                  xyaxes=values(camera["front_xyaxes"]), fovy=str(camera["fovy"]))
    # Imported robot-local wrist cameras retain their source approximation.
    # Missing cameras fail explicitly in the environment; never fabricate a
    # camera and silently present it as a calibrated robot sensor.
    return ET.tostring(root, encoding="unicode")
