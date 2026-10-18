import os
import json
import math
import argparse
import numpy as np

try:
    from ai2thor.controller import Controller
except ImportError:
    print("ai2thor not found. Please install it.")
    exit(1)

from eval_connectivity import load_scene, edt_spawn


def is_clear(x, z, furniture_boxes, l, w):
    clearance = 0.3
    for box in furniture_boxes:
        min_x, max_x, min_z, max_z = box
        if (min_x - clearance <= x <= max_x + clearance) and (min_z - clearance <= z <= max_z + clearance):
            return False
    if x < clearance or x > l - clearance or z < clearance or z > w - clearance:
        return False
    return True


def legacy_spawn(furniture_boxes, l, w):
    """Pre-2026-10 behaviour: first 0.25 m grid cell 0.3 m clear of every AABB (falls back to centre)."""
    for x in np.arange(0.25, l, 0.25):
        for z in np.arange(0.25, w, 0.25):
            if is_clear(x, z, furniture_boxes, l, w):
                return x, z
    return l / 2.0, w / 2.0


def save_results(results, save_path, mode):
    total_objects_evaluated = sum(scene['total_objects'] for scene in results.values())
    total_objects_accessible = sum(scene['accessible_objects'] for scene in results.values())
    avg_accessibility = total_objects_accessible / total_objects_evaluated if total_objects_evaluated > 0 else 0
    rates = [s['accessibility_rate'] for s in results.values() if s['total_objects'] > 0]
    with open(save_path, 'w') as f:
        json.dump({
            "summary": {
                "mode": mode,
                "instance_count": len(results),
                "failed_scenes": sum(1 for s in results.values() if s.get('status', 'ok') != 'ok'),
                "total_objects_evaluated": total_objects_evaluated,
                "total_objects_accessible": total_objects_accessible,
                "average_accessibility_rate": avg_accessibility,
                "per_scene_mean_accessibility": float(np.mean(rates)) if rates else 0,
            },
            "scenes": results
        }, f, indent=2)
    return total_objects_evaluated, total_objects_accessible, avg_accessibility


def evaluate_navigation(scenes_dir, resume=False, legacy=False, out_name=None):
    json_files = sorted(f for f in os.listdir(scenes_dir) if f.endswith('.json') and not f.startswith('._') and not f.startswith(('walkability_', 'navigation_', 'connectivity_')))

    if not json_files:
        print(f"No JSON scene files found in {scenes_dir}")
        return

    controller = Controller(agentMode="default", visibilityDistance=1.5, scene="Procedural", gridSize=0.25)

    mode = "legacy" if legacy else "edt_spawn_fail_zero"
    save_path = os.path.join(scenes_dir, out_name or ("navigation_results.json" if legacy else "navigation_results_v2.json"))
    results = {}
    if resume and os.path.exists(save_path):
        with open(save_path, 'r') as f:
            results = json.load(f).get("scenes", {})
        print(f"Resuming from checkpoint: {len(results)} scenes already evaluated.")

    print(f"Found {len(json_files)} scenes to evaluate for Navigation Accessibility ({mode}).")

    for filename in json_files:
        scene_name = filename.replace('.json', '')
        if scene_name in results:
            continue
        file_path = os.path.join(scenes_dir, filename)

        with open(file_path, 'r') as f:
            house_data = json.load(f)

        rd = house_data.get('room_dims', {})
        l = rd.get('l', 0)
        w = rd.get('w', 0)
        furniture_boxes = house_data.get('furniture_boxes', [])

        def fail(status, spawn=None):
            # v2: a scene the agent cannot navigate counts as 0 accessible objects instead of vanishing
            results[scene_name] = {
                "total_objects": len(furniture_boxes),
                "accessible_objects": 0,
                "accessibility_rate": 0.0,
                "status": status,
                "spawn": spawn,
            }
            save_results(results, save_path, mode)

        try:
            controller.reset(scene=house_data)
        except Exception as e:
            print(f"Failed to load {scene_name}: {e}")
            if not legacy:
                fail("load_failed")
            continue

        if legacy:
            spawn_x, spawn_z = legacy_spawn(furniture_boxes, l, w)
            spawn_clearance = None
        else:
            # EDT maximum of the rasterised free floor (exact rotated footprints), snapped to the 0.25 m grid
            _, _, obstacles = load_scene(file_path)
            sp = edt_spawn(l, w, [o for o in obstacles if o['blocking']])
            spawn_x, spawn_z, spawn_clearance = sp if sp else (l / 2.0, w / 2.0, 0.0)

        controller.step(action="Teleport", position={"x": spawn_x, "y": 0.9, "z": spawn_z}, forceAction=True)
        spawn = [float(spawn_x), float(spawn_z)]

        rp_event = controller.step(action="GetReachablePositions")
        if not rp_event.metadata["lastActionSuccess"]:
            if not legacy:
                fail("reachable_positions_failed", spawn)
            continue

        reachable_positions = rp_event.metadata["actionReturn"]
        if not reachable_positions:
            if not legacy:
                fail("no_reachable_positions", spawn)
            continue

        scene_objects = 0
        scene_accessible = 0

        # Test accessibility to every object
        for idx, box in enumerate(furniture_boxes):
            min_x, max_x, min_z, max_z = box

            # An object is accessible if there is a reachable NavMesh point within 0.75m of its edge
            is_accessible = False
            for p in reachable_positions:
                px, pz = p['x'], p['z']
                # Distance to the bounding box
                dx = max(min_x - px, 0, px - max_x)
                dz = max(min_z - pz, 0, pz - max_z)
                dist_to_box = math.hypot(dx, dz)

                if dist_to_box <= 0.75:
                    is_accessible = True
                    break

            scene_objects += 1
            if is_accessible:
                scene_accessible += 1

        if scene_objects > 0:
            results[scene_name] = {
                "total_objects": scene_objects,
                "accessible_objects": scene_accessible,
                "accessibility_rate": scene_accessible / scene_objects,
                "status": "ok",
                "spawn": spawn,
                "spawn_clearance": spawn_clearance,
                "n_reachable_positions": len(reachable_positions),
            }
            save_results(results, save_path, mode)

    controller.stop()

    total_objects_evaluated, total_objects_accessible, avg_accessibility = save_results(results, save_path, mode)

    print(f"\nNavigation Evaluation Complete for {scenes_dir}! Total rooms evaluated: {len(results)}")
    print(f"Overall Accessibility Rate: {avg_accessibility * 100:.2f}% ({total_objects_accessible}/{total_objects_evaluated} objects)")
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Navigation/Accessibility for ProcTHOR JSON scenes.")
    parser.add_argument("--scenes_dir", type=str, required=True, help="Directory containing the converted JSON files")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing results checkpoint")
    parser.add_argument("--legacy", action="store_true",
                        help="Reproduce the pre-2026-10 numbers: first-clear-cell spawn, scenes where "
                             "GetReachablePositions fails are skipped (writes navigation_results.json)")
    parser.add_argument("--out_name", type=str, default=None,
                        help="Output filename inside scenes_dir (default: navigation_results_v2.json, "
                             "or navigation_results.json with --legacy)")
    args = parser.parse_args()

    evaluate_navigation(args.scenes_dir, resume=args.resume, legacy=args.legacy, out_name=args.out_name)
