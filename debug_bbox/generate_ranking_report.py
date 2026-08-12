import os
import glob
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.drawing.image import Image as OpenPyxlImage

def natural_sort_key(s):
    """Sorts strings using natural numerical ordering (matching VS Code order)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def load_relationship_database():
    """Loads and indexes scene graph relationship JSON files from FRONT directory."""
    scans_rel_map = {}
    front_dir = "/Users/lehoangan/Documents/GitHub/ROOM/echoscene/FRONT"
    json_files = glob.glob(os.path.join(front_dir, "relationships_*.json"))
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
                for scan in data.get("scans", []):
                    sid = scan.get("scan")
                    if sid:
                        scans_rel_map[sid] = scan
        except Exception:
            pass
    return scans_rel_map

def get_obb_polygon(x, z, l, w, angle_deg):
    """Calculates 2D Shapely Polygon for Oriented Bounding Box."""
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    corners = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
    rotated = []
    for cx, cz in corners:
        rx = cx * cos_a - cz * sin_a
        rz = cx * sin_a + cz * cos_a
        rotated.append((x + rx, z + rz))
    return Polygon(rotated)

def parse_log_file(file_path):
    """Parses raw object bounding box details, scene_id, and pairwise overlap logs."""
    objects = []
    raw_overlaps = []
    scene_id = None
    if not os.path.exists(file_path):
        return objects, raw_overlaps, scene_id
        
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    start_parsing = False
    for line in lines:
        if "SCENE:" in line:
            match = re.search(r'SCENE:\s*([\w-]+)', line)
            if match:
                scene_id = match.group(1)
        if "---" in line:
            start_parsing = True
            continue
        if "Pairwise overlap" in line or "==" in line:
            start_parsing = False
        if "!! OVERLAP:" in line:
            raw_overlaps.append(line.strip())
            
        if start_parsing and line.strip():
            parts = line.split()
            if len(parts) >= 8:
                name = parts[0]
                try:
                    l = float(parts[1])
                    h = float(parts[2])
                    w = float(parts[3])
                    x = float(parts[4])
                    y = float(parts[5])
                    z = float(parts[6])
                    angle = float(parts[7].replace('°', ''))
                    objects.append({
                        "name": name, "l": l, "h": h, "w": w,
                        "x": x, "y": y, "z": z, "angle": angle
                    })
                except ValueError:
                    continue
    return objects, raw_overlaps, scene_id

def evaluate_relationship_accuracy(objects, scene_id, scans_rel_map):
    """Evaluates spatial & attribute relationship accuracy against GT scene graph."""
    if not scene_id or scene_id not in scans_rel_map:
        return {
            "rel_acc": 0.0, "total_triplets": 0, "lr_acc": 0.0,
            "fb_acc": 0.0, "stand_acc": 0.0, "close_acc": 0.0
        }
        
    scan_info = scans_rel_map[scene_id]
    rel_triples = scan_info.get("relationships", [])
    json_objs = scan_info.get("objects", {})
    
    # Map 1-based JSON object key to index in objects list
    sorted_keys = sorted(list(json_objs.keys()), key=lambda k: int(k) if k.isdigit() else k)
    key_to_obj = {}
    for i, k in enumerate(sorted_keys):
        if i < len(objects):
            key_to_obj[int(k)] = objects[i]
            
    accuracies = {k: [] for k in ["left", "right", "front", "behind", "smaller", "bigger", "shorter", "taller", "standing on", "close by", "symmetrical to", "total"]}
    
    for rel in rel_triples:
        s_id, o_id, r_id, r_name = rel[0], rel[1], rel[2], rel[3].strip().lower()
        if s_id not in key_to_obj or o_id not in key_to_obj:
            continue
        obj_s = key_to_obj[s_id]
        obj_o = key_to_obj[o_id]
        
        box_s = [obj_s["l"], obj_s["h"], obj_s["w"], obj_s["x"], obj_s["y"], obj_s["z"]]
        box_o = [obj_o["l"], obj_o["h"], obj_o["w"], obj_o["x"], obj_o["y"], obj_o["z"]]
        
        if r_name == "left":
            ok = (box_s[5] - box_o[5] <= -0.05)
        elif r_name == "right":
            ok = (box_s[5] - box_o[5] >= 0.05)
        elif r_name == "front":
            ok = (box_s[3] - box_o[3] >= -0.05)
        elif r_name == "behind":
            ok = (box_s[3] - box_o[3] <= 0.05)
        elif r_name in ["bigger than", "bigger"]:
            v_s = box_s[0]*box_s[1]*box_s[2]
            v_o = box_o[0]*box_o[1]*box_o[2]
            ok = ((v_s - v_o)/max(v_s, 1e-5) >= 0.15)
        elif r_name in ["smaller than", "smaller"]:
            v_s = box_s[0]*box_s[1]*box_s[2]
            v_o = box_o[0]*box_o[1]*box_o[2]
            ok = ((v_s - v_o)/max(v_s, 1e-5) <= -0.15)
        elif r_name in ["taller than", "taller"]:
            h_s = box_s[4] + box_s[1]
            h_o = box_o[4] + box_o[1]
            ok = ((h_s - h_o)/max(h_s, 1e-5) >= 0.1)
        elif r_name in ["shorter than", "shorter"]:
            h_s = box_s[4] + box_s[1]
            h_o = box_o[4] + box_o[1]
            ok = ((h_s - h_o)/max(h_s, 1e-5) <= -0.1)
        elif r_name in ["standing on", "above"]:
            # Original formula as requested: np.abs(box_s[4] - box_o[4]) < 0.04
            ok = (abs(box_s[4] - box_o[4]) < 0.04)
        elif r_name == "close by":
            dist = np.sqrt((box_s[3]-box_o[3])**2 + (box_s[4]-box_o[4])**2 + (box_s[5]-box_o[5])**2)
            ok = (dist <= 0.45)
        elif r_name == "symmetrical to":
            dist = np.sqrt((-box_s[3]-box_o[3])**2 + (-box_s[5]-box_o[5])**2)
            ok = (dist <= 0.45)
        else:
            continue
            
        val = 1 if ok else 0
        if r_name in accuracies:
            accuracies[r_name].append(val)
        accuracies["total"].append(val)
        
    def calc_mean(lst):
        return float(np.mean(lst)) * 100.0 if lst and len(lst) > 0 else 0.0
        
    lr_acc = (calc_mean(accuracies["left"]) + calc_mean(accuracies["right"])) / 2.0 if (accuracies["left"] or accuracies["right"]) else 0.0
    fb_acc = (calc_mean(accuracies["front"]) + calc_mean(accuracies["behind"])) / 2.0 if (accuracies["front"] or accuracies["behind"]) else 0.0
    stand_acc = calc_mean(accuracies["standing on"])
    close_acc = calc_mean(accuracies["close by"])
    tot_acc = calc_mean(accuracies["total"])
    
    return {
        "rel_acc": round(tot_acc, 1),
        "total_triplets": len(accuracies["total"]),
        "lr_acc": round(lr_acc, 1),
        "fb_acc": round(fb_acc, 1),
        "stand_acc": round(stand_acc, 1),
        "close_acc": round(close_acc, 1)
    }

def analyze_scene(objects, raw_overlaps, scene_id, scans_rel_map):
    """Computes layout quality metrics (collisions, OBB overlap area, out-of-bounds, walkability, relationship accuracy)."""
    # Separate floor and furniture
    floor_obj = None
    furniture_objs = []
    for obj in objects:
        if obj["name"] == "floor":
            floor_obj = obj
        elif obj["name"] not in ["_scene_", "lamp"]:
            furniture_objs.append(obj)
            
    # Define Floor Polygon
    if floor_obj is not None:
        floor_poly = get_obb_polygon(floor_obj["x"], floor_obj["z"], floor_obj["l"], floor_obj["w"], floor_obj["angle"])
    else:
        floor_poly = Polygon([[-2.0, -2.5], [2.0, -2.5], [2.0, 2.5], [-2.0, 2.5]])
        
    # Furniture OBB Polygons
    furn_polys = []
    for obj in furniture_objs:
        poly = get_obb_polygon(obj["x"], obj["z"], obj["l"], obj["w"], obj["angle"])
        furn_polys.append((obj, poly))
        
    # 1. Furniture-Furniture AABB Overlaps (from log)
    furn_aabb_overlaps = 0
    for line in raw_overlaps:
        match = re.search(r'OVERLAP:\s*(\w+)\s*<->\s*(\w+)', line)
        if match:
            o1, o2 = match.group(1), match.group(2)
            if o1 not in ["floor", "lamp", "_scene_"] and o2 not in ["floor", "lamp", "_scene_"]:
                furn_aabb_overlaps += 1
                
    # 2. Exact 2D OBB Overlap Area & Pairwise Intersection
    total_obb_overlap_area = 0.0
    obb_overlap_count = 0
    for i in range(len(furn_polys)):
        for j in range(i + 1, len(furn_polys)):
            p1 = furn_polys[i][1]
            p2 = furn_polys[j][1]
            if p1.intersects(p2):
                inter = p1.intersection(p2)
                area = inter.area
                if area > 1e-4:
                    total_obb_overlap_area += area
                    obb_overlap_count += 1
                    
    # 3. Out of Bounds (Protrusion past room floor)
    oob_count = 0
    max_oob_dist = 0.0
    for obj, poly in furn_polys:
        if not floor_poly.contains(poly):
            oob_count += 1
            coords = np.array(poly.exterior.coords)
            for cx, cz in coords:
                pt_dist = floor_poly.distance(Point(cx, cz))
                if pt_dist > max_oob_dist:
                    max_oob_dist = pt_dist

    # 4. Free Floor & Walkable Connected Components
    if furn_polys:
        all_furn_union = unary_union([p for _, p in furn_polys])
        free_floor = floor_poly.difference(all_furn_union)
    else:
        free_floor = floor_poly
        
    free_floor_area = free_floor.area if free_floor else 0.0
    floor_total_area = floor_poly.area if floor_poly.area > 0 else 1.0
    free_floor_pct = (free_floor_area / floor_total_area) * 100.0
    
    if isinstance(free_floor, Polygon):
        walkable_components = 1 if free_floor.area > 0.01 else 0
    elif isinstance(free_floor, MultiPolygon):
        walkable_components = sum(1 for p in free_floor.geoms if p.area > 0.05)
    else:
        walkable_components = 0
        
    # 5. Center Clearance (Distance from (0,0) to nearest furniture)
    center_pt = Point(0, 0)
    min_center_dist = 999.0
    for _, poly in furn_polys:
        d = poly.distance(center_pt)
        if d < min_center_dist:
            min_center_dist = d
    if min_center_dist == 999.0:
        min_center_dist = 0.0

    # 6. Relationship Accuracy
    rel_metrics = evaluate_relationship_accuracy(objects, scene_id, scans_rel_map)

    # 7. Overall Quality Score (0 to 100)
    score = 100.0
    score -= furn_aabb_overlaps * 20.0
    score -= total_obb_overlap_area * 35.0
    score -= oob_count * 15.0
    score -= max_oob_dist * 25.0
    score -= max(0, walkable_components - 1) * 8.0
    score += min(10.0, free_floor_pct * 0.15) # Small reward for open space
    score = float(np.clip(score, 0.0, 100.0))
    
    # Status / Grade
    if score >= 90.0:
        status = "EXCELLENT"
    elif score >= 75.0:
        status = "GOOD"
    elif score >= 50.0:
        status = "MINOR ISSUES"
    else:
        status = "SEVERE COLLISIONS"
        
    return {
        "score": score,
        "status": status,
        "furn_aabb_overlaps": furn_aabb_overlaps,
        "obb_overlap_count": obb_overlap_count,
        "total_obb_overlap_area": round(total_obb_overlap_area, 4),
        "oob_count": oob_count,
        "max_oob_dist": round(max_oob_dist, 4),
        "free_floor_pct": round(free_floor_pct, 1),
        "walkable_components": walkable_components,
        "center_clearance": round(min_center_dist, 3),
        "rel_acc": rel_metrics["rel_acc"],
        "total_triplets": rel_metrics["total_triplets"],
        "lr_acc": rel_metrics["lr_acc"],
        "fb_acc": rel_metrics["fb_acc"],
        "stand_acc": rel_metrics["stand_acc"],
        "close_acc": rel_metrics["close_acc"]
    }

def render_scene_image(file_path, objects, metrics, out_img_path):
    """Renders high-quality 2D top-down layout plot image with metric overlay."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=150)
    
    color_map = {
        "bed": "#3498db", "chair": "#e74c3c", "nightstand": "#9b59b6", 
        "table": "#e67e22", "wardrobe": "#2ecc71", "lamp": "#f1c40f", 
        "floor": "#ecf0f1"
    }
    
    # Draw floor first
    floor_obj = next((o for o in objects if o["name"] == "floor"), None)
    if floor_obj:
        f_poly = get_obb_polygon(floor_obj["x"], floor_obj["z"], floor_obj["l"], floor_obj["w"], floor_obj["angle"])
        f_patch = patches.Polygon(np.array(f_poly.exterior.coords), closed=True, facecolor="#f8f9fa", edgecolor="#7f8c8d", linewidth=2.0)
        ax.add_patch(f_patch)
        
    # Draw furniture OBBs
    for obj in objects:
        if obj["name"] in ["_scene_", "floor"]:
            continue
        poly = get_obb_polygon(obj["x"], obj["z"], obj["l"], obj["w"], obj["angle"])
        c = color_map.get(obj["name"], "#1abc9c")
        alpha = 0.7 if obj["name"] != "lamp" else 0.4
        
        patch = patches.Polygon(np.array(poly.exterior.coords), closed=True, facecolor=c, edgecolor="#2c3e50", alpha=alpha, linewidth=1.5)
        ax.add_patch(patch)
        
        # Label
        if obj["name"] != "lamp":
            ax.text(obj["x"], obj["z"], obj["name"], ha='center', va='center', fontsize=8, weight='bold',
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.5))
                    
    # View settings
    ax.set_aspect('equal')
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.grid(True, linestyle=':', alpha=0.4)
    
    # Title & Badge
    score = metrics["score"]
    status = metrics["status"]
    rel_acc = metrics["rel_acc"]
    
    fname = os.path.basename(file_path)
    if len(fname) > 45:
        fname = fname[:42] + "..."
    ax.set_title(f"{fname}\nScore: {score:.1f}/100 [{status}] | Rel Acc: {rel_acc:.1f}%", fontsize=8.5, weight='bold', color='black', pad=6)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
    plt.savefig(out_img_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def build_excel_report(ranked_data, output_excel):
    """Creates a beautifully formatted Excel report with embedded layout images and Relationship Accuracy."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Layout Rankings Summary"
    ws.views.sheetView[0].showGridLines = True

    # Colors
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid") # Dark Navy
    sub_header_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid") # Soft Blue
    zebra_fill = PatternFill(start_color="F9FAFC", end_color="F9FAFC", fill_type="solid")
    
    font_title = Font(name="Segoe UI", size=16, bold=True, color="FFFFFF")
    font_header = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
    font_body = Font(name="Segoe UI", size=10)
    font_bold = Font(name="Segoe UI", size=10, bold=True)
    
    border_thin = Border(
        left=Side(style='thin', color='D9D9D9'), right=Side(style='thin', color='D9D9D9'),
        top=Side(style='thin', color='D9D9D9'), bottom=Side(style='thin', color='D9D9D9')
    )

    # 1. Title Banner
    ws.merge_cells("A1:Q2")
    cell_title = ws["A1"]
    cell_title.value = "  ROOM Layout Quality, Walkability & Relationship Benchmark Report"
    cell_title.font = font_title
    cell_title.fill = header_fill
    cell_title.alignment = Alignment(vertical="center", horizontal="left")
    
    # 2. Subtitle / Summary info
    ws.merge_cells("A3:Q3")
    cell_sub = ws["A3"]
    cell_sub.value = f"  Total Runs Analyzed: {len(ranked_data)}  |  Metrics: Relationship Accuracy (L/R, F/B, Stand, Close), Overlaps, OBB Area, Out-of-Bounds, Free Floor %, Walkability"
    cell_sub.font = Font(name="Segoe UI", size=10, italic=True, color="1F4E78")
    cell_sub.fill = sub_header_fill
    cell_sub.alignment = Alignment(vertical="center", horizontal="left")
    
    # 3. Column Headers
    headers = [
        "Rank", "Layout Visualization", "Log File Name", "Quality Score",
        "Rel Acc (%)", "L/R Acc (%)", "F/B Acc (%)", "Stand Acc (%)", "Close Acc (%)",
        "Overlaps (Count)", "OBB Overlap (m²)", "Out of Bounds", "Max OOB Dist (m)",
        "Free Floor %", "Walkable Components", "Center Clear (m)", "Grade Status"
    ]
    
    ws.row_dimensions[5].height = 28
    for col_num, h_text in enumerate(headers, 1):
        cell = ws.cell(row=5, column=col_num)
        cell.value = h_text
        cell.font = font_header
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border_thin

    # Column widths
    col_widths = {
        "A": 8,   # Rank
        "B": 24,  # Image
        "C": 48,  # File name
        "D": 15,  # Quality Score
        "E": 14,  # Rel Acc %
        "F": 13,  # L/R Acc %
        "G": 13,  # F/B Acc %
        "H": 13,  # Stand Acc %
        "I": 13,  # Close Acc %
        "J": 16,  # Overlaps
        "K": 18,  # OBB Overlap area
        "L": 15,  # OOB count
        "M": 18,  # Max OOB dist
        "N": 14,  # Free Floor %
        "O": 20,  # Walkable components
        "P": 16,  # Center clearance
        "Q": 20   # Status
    }
    for col_letter, width in col_widths.items():
        ws.column_dimensions[col_letter].width = width

    # 4. Fill Data Rows
    start_row = 6
    for idx, item in enumerate(ranked_data, 1):
        row_num = start_row + idx - 1
        ws.row_dimensions[row_num].height = 115 # Fit thumbnail cleanly
        
        m = item["metrics"]
        fname = os.path.basename(item["file_path"])
        img_path = item["img_path"]
        
        # Rank
        cell_rank = ws.cell(row=row_num, column=1, value=f"#{idx}")
        cell_rank.alignment = Alignment(horizontal="center", vertical="center")
        cell_rank.font = font_bold
        
        # Image
        if os.path.exists(img_path):
            img = OpenPyxlImage(img_path)
            img.width = 140
            img.height = 140
            ws.add_image(img, f"B{row_num}")
            
        # File name
        cell_file = ws.cell(row=row_num, column=3, value=fname)
        cell_file.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        cell_file.font = font_body
        
        # Score
        score_val = m["score"]
        cell_score = ws.cell(row=row_num, column=4, value=score_val)
        cell_score.alignment = Alignment(horizontal="center", vertical="center")
        cell_score.font = Font(name="Segoe UI", size=11, bold=True)
        cell_score.number_format = '0.0'
        
        if score_val >= 90:
            cell_score.fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid") # Light Green
        elif score_val >= 75:
            cell_score.fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid") # Soft Yellow
        elif score_val >= 50:
            cell_score.fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid") # Light Orange
        else:
            cell_score.fill = PatternFill(start_color="F8CBAD", end_color="F8CBAD", fill_type="solid") # Light Red
            
        # Relationship Accuracy Metrics
        cell_rel = ws.cell(row=row_num, column=5, value=m["rel_acc"])
        cell_rel.alignment = Alignment(horizontal="center", vertical="center")
        cell_rel.font = font_bold
        
        ws.cell(row=row_num, column=6, value=m["lr_acc"]).alignment = Alignment(horizontal="center", vertical="center")
        ws.cell(row=row_num, column=7, value=m["fb_acc"]).alignment = Alignment(horizontal="center", vertical="center")
        ws.cell(row=row_num, column=8, value=m["stand_acc"]).alignment = Alignment(horizontal="center", vertical="center")
        ws.cell(row=row_num, column=9, value=m["close_acc"]).alignment = Alignment(horizontal="center", vertical="center")
        
        # Layout Metrics
        ws.cell(row=row_num, column=10, value=m["furn_aabb_overlaps"]).alignment = Alignment(horizontal="center", vertical="center")
        ws.cell(row=row_num, column=11, value=m["total_obb_overlap_area"]).alignment = Alignment(horizontal="center", vertical="center")
        ws.cell(row=row_num, column=12, value=m["oob_count"]).alignment = Alignment(horizontal="center", vertical="center")
        ws.cell(row=row_num, column=13, value=m["max_oob_dist"]).alignment = Alignment(horizontal="center", vertical="center")
        ws.cell(row=row_num, column=14, value=m["free_floor_pct"]).alignment = Alignment(horizontal="center", vertical="center")
        ws.cell(row=row_num, column=15, value=m["walkable_components"]).alignment = Alignment(horizontal="center", vertical="center")
        ws.cell(row=row_num, column=16, value=m["center_clearance"]).alignment = Alignment(horizontal="center", vertical="center")
        
        # Status Grade Badge
        cell_status = ws.cell(row=row_num, column=17, value=m["status"])
        cell_status.alignment = Alignment(horizontal="center", vertical="center")
        cell_status.font = font_bold
        
        if m["status"] == "EXCELLENT":
            cell_status.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
            cell_status.font = Font(name="Segoe UI", size=10, bold=True, color="006100")
        elif m["status"] == "GOOD":
            cell_status.fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
            cell_status.font = Font(name="Segoe UI", size=10, bold=True, color="375623")
        elif m["status"] == "MINOR ISSUES":
            cell_status.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
            cell_status.font = Font(name="Segoe UI", size=10, bold=True, color="9C6500")
        else:
            cell_status.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
            cell_status.font = Font(name="Segoe UI", size=10, bold=True, color="9C0006")
            
        # Set borders & fonts for row cells
        for c in range(1, 18):
            cell_item = ws.cell(row=row_num, column=c)
            cell_item.border = border_thin
            if c not in [4, 5, 17]:
                cell_item.font = font_body
            if idx % 2 == 0 and c not in [4, 5, 17]:
                cell_item.fill = zebra_fill

    wb.save(output_excel)
    print(f"Excel report successfully generated: {output_excel}")

def main():
    debug_dir = "/Users/lehoangan/Documents/GitHub/ROOM/echoscene/debug_bbox"
    render_dir = os.path.join(debug_dir, "rendered_plots")
    excel_file = os.path.join(debug_dir, "layout_ranking_report.xlsx")
    
    scans_rel_map = load_relationship_database()
    print(f"Loaded relationship database for {len(scans_rel_map)} scenes.")
    
    files = sorted(glob.glob(os.path.join(debug_dir, "*.txt")), key=natural_sort_key)
    print(f"Analyzing {len(files)} log files...")
    
    scenarios = []
    for fpath in files:
        objs, overlaps, scene_id = parse_log_file(fpath)
        if not objs:
            continue
        metrics = analyze_scene(objs, overlaps, scene_id, scans_rel_map)
        
        fname = os.path.basename(fpath).replace(".txt", "")
        img_out = os.path.join(render_dir, f"{fname}.png")
        render_scene_image(fpath, objs, metrics, img_out)
        
        scenarios.append({
            "file_path": fpath,
            "metrics": metrics,
            "img_path": img_out
        })
        
    # Rank scenarios by Quality Score (Descending)
    scenarios.sort(key=lambda x: x["metrics"]["score"], reverse=True)
    
    print("\n" + "="*80)
    print(" TOP 5 BEST LAYOUT CONFIGURATIONS ")
    print("="*80)
    for rank, item in enumerate(scenarios[:5], 1):
        m = item["metrics"]
        print(f"Rank #{rank}: {os.path.basename(item['file_path'])}")
        print(f"   Score: {m['score']:.1f}/100 [{m['status']}] | Rel Acc: {m['rel_acc']:.1f}% | Overlaps: {m['furn_aabb_overlaps']} | OBB Area: {m['total_obb_overlap_area']} m²")
    print("="*80)
    
    build_excel_report(scenarios, excel_file)

if __name__ == "__main__":
    main()
