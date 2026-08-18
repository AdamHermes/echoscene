import os

filepath = 'scripts/eval_3dssg.py'
with open(filepath, 'r') as f:
    lines = f.readlines()

new_lines = []
in_main = False
for i, line in enumerate(lines):
    if "parser.add_argument('--json_path'" in line:
        new_lines.append("    parser.add_argument('--json_path', required=False, type=str, default=None, help='Path to adapted 3DSSG JSON')\n")
        new_lines.append("    parser.add_argument('--json_dir', required=False, type=str, default=None, help='Path to directory of JSONs')\n")
        new_lines.append("    parser.add_argument('--start_idx', type=int, default=0, help='Start index')\n")
        new_lines.append("    parser.add_argument('--max_samples', type=int, default=-1, help='Max samples')\n")
    elif "cond_model, _ = clip.load" in line and "Extracting CLIP embeddings" not in lines[i-1]:
        pass # Skip
    elif 'print(f"Loading 3DSSG data from' in line:
        new_lines.append('''
    if args.json_path:
        json_files = [args.json_path]
    elif args.json_dir:
        import glob
        json_files = sorted(glob.glob(os.path.join(args.json_dir, '*.json')))
        if args.start_idx > 0:
            json_files = json_files[args.start_idx:]
        if args.max_samples > 0:
            json_files = json_files[:args.max_samples]
    else:
        print('Error: provide --json_path or --json_dir')
        return

    if modelArgs.get('with_CLIP', True):
        print('Loading CLIP model once...')
        cond_model, _ = clip.load('ViT-B/32', device='cuda')

    for file_idx, json_path in enumerate(json_files):
        print(f'\\n--- Processing {file_idx+1}/{len(json_files)}: {json_path} ---')
''')
        new_lines.append("        with open(json_path, 'r') as f:\n")
        in_main = True
    elif in_main and line.startswith('if __name__ == '):
        in_main = False
        new_lines.append(line)
    elif in_main:
        if line.strip() == '':
            new_lines.append(line)
        elif "with open(args.json_path, 'r') as f:" in line:
            pass # Skip original open
        elif "scene_data = json.load(f)" in line and "with open(args.json_path" in lines[i-1]:
            new_lines.append("            scene_data = json.load(f)\n")
        elif 'cond_model, _ = clip.load("ViT-B/32"' in line:
            pass # skip loading inside loop
        elif 'print("Extracting CLIP embeddings...")' in line:
            pass
        else:
            new_lines.append("    " + line)
    else:
        new_lines.append(line)

with open(filepath, 'w') as f:
    f.writelines(new_lines)
print("eval_3dssg.py patched successfully!")
