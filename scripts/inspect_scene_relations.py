from __future__ import print_function
import argparse, json, sys
from pathlib import Path
from collections import Counter

SCRIPT_DIR   = Path(sys.argv[0]).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"
GRAY   = "\033[90m"
WHITE  = "\033[97m"
NO_COLOR = False

def c(code, txt):
    if NO_COLOR: return txt
    return code + txt + RESET

ROOM_TYPE_TO_FILE = {
    "bedroom":    "relationships_bedroom_test.json",
    "livingroom": "relationships_livingroom_test.json",
    "diningroom": "relationships_diningroom_test.json",
    "library":    "relationships_library_all.json",
    "all":        "relationships_all_test.json",
}

SPATIAL_DIR  = {"left", "right", "front", "behind"}
SPATIAL_VERT = {"above", "standing on"}
SIZE_RELS    = {"bigger than", "smaller than", "taller than", "shorter than"}
STYLE_RELS   = {"same style as", "same super category as", "same material as",
                "symmetrical to", "close by"}

def rel_col(r):
    r = r.lower()
    if r in SPATIAL_DIR:  return CYAN
    if r in SPATIAL_VERT: return YELLOW
    if r in SIZE_RELS:    return RED
    if r in STYLE_RELS:   return GRAY
    return WHITE

def print_scene(scan, filter_rel="", filter_obj=""):
    name = scan["scan"]
    objs = scan["objects"]
    rels = scan["relationships"]
    sep  = "=" * 60
    dash = "-" * 60
    print("\n" + c(BOLD+WHITE, sep))
    print(c(BOLD+WHITE, "  Scene : ") + c(GREEN, name))
    print(c(BOLD+WHITE, "  Objects (" + str(len(objs)) + ")"))
    print(c(WHITE, dash))
    for oid, oname in sorted(objs.items(), key=lambda x: int(x[0])):
        print("  " + c(YELLOW, "[" + oid.rjust(2) + "]") + "  " + oname)
    print("\n" + c(BOLD+WHITE, "  Relations (" + str(len(rels)) + " total)"))
    print(c(WHITE, dash))
    printed = 0
    for rel in rels:
        sub_id, obj_id, rel_idx, rel_name = rel[0], rel[1], rel[2], rel[3]
        sub_name = objs.get(str(sub_id), "?" + str(sub_id))
        obj_name = objs.get(str(obj_id), "?" + str(obj_id))
        if filter_rel and filter_rel.lower() not in rel_name.lower():
            continue
        if filter_obj:
            fo = filter_obj.lower()
            if fo not in sub_name.lower() and fo not in obj_name.lower():
                continue
        row = ("  " + c(YELLOW, sub_name.ljust(22))
               + "  " + c(rel_col(rel_name), rel_name.ljust(22))
               + "  " + c(CYAN, obj_name)
               + "  " + c(GRAY, "(rel_idx=" + str(rel_idx) + ")"))
        print(row)
        printed += 1
    if printed == 0:
        print("  " + c(RED, "(khong co relation nao khop filter)"))
    print(c(WHITE, sep) + "\n")

def print_summary(scan):
    rels = scan["relationships"]
    counter = Counter(r[3] for r in rels)
    print("\n" + c(BOLD+WHITE, "  Thong ke relation:"))
    print(c(WHITE, "-" * 40))
    for rel_name, cnt in sorted(counter.items(), key=lambda x: -x[1]):
        print("  " + c(rel_col(rel_name), rel_name.ljust(26)) + "  " + c(BOLD, str(cnt).rjust(3)))
    print(c(WHITE, "-" * 40))
    print("  " + c(BOLD, "Total: " + str(len(rels))) + "\n")

def main():
    global NO_COLOR
    p = argparse.ArgumentParser(
        description="Inspect relations cua mot scene trong tap TEST 3D-FRONT."
    )
    p.add_argument("--scene_id",   required=True,
                   help="ID scene can tim, vi du: 6482")
    p.add_argument("--room_type",  default="bedroom",
                   choices=["bedroom","livingroom","diningroom","library","all"],
                   help="Loai phong (default: bedroom)")
    p.add_argument("--json_file",  default=None,
                   help="Chi dinh truc tiep file JSON (ghi de --room_type)")
    p.add_argument("--filter_rel", default="",
                   help="Loc theo ten relation (vi du: left, above)")
    p.add_argument("--filter_obj", default="",
                   help="Loc theo ten object (vi du: bed, wardrobe)")
    p.add_argument("--no_color",   action="store_true",
                   help="Tat mau ANSI")
    p.add_argument("--data_dir",   default=None,
                   help="Thu muc chua FRONT/ (default: thu muc goc project)")
    args = p.parse_args()

    if args.no_color:
        NO_COLOR = True

    if args.json_file:
        json_path = Path(args.json_file)
    else:
        data_dir  = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "FRONT"
        json_path = data_dir / ROOM_TYPE_TO_FILE[args.room_type]

    if not json_path.exists():
        print(c(RED, "[ERROR] Khong tim thay file: " + str(json_path)))
        sys.exit(1)

    print(c(GRAY, "Doc file: " + str(json_path)))
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    matches = [s for s in data["scans"] if args.scene_id in s["scan"]]
    if not matches:
        print(c(RED, "[ERROR] Khong tim thay scene " + repr(args.scene_id)
                + " trong " + json_path.name))
        cands = [s["scan"] for s in data["scans"] if args.scene_id[:3] in s["scan"]][:5]
        if cands:
            print(c(YELLOW, "Goi y scene gan dung:"))
            for cc in cands:
                print("  " + cc)
        sys.exit(1)

    for scan in matches:
        print_scene(scan, filter_rel=args.filter_rel, filter_obj=args.filter_obj)
        print_summary(scan)

if __name__ == "__main__":
    main()
