import open3d as o3d
import argparse
import os
import json
import numpy as np
import torch
import clip
import sys
from pathlib import Path
from omegaconf import OmegaConf

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from model.SGDiff import SGDiff
from helpers.util import bool_flag, descale_box_params, postprocess_sincos2arctan
from helpers.visualize_scene import render_full
from helpers.structured_scene_export import export_structured_scene

def evaluate_3dssg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', required=True, type=str, help="Path to adapted 3DSSG JSON")
    parser.add_argument('--dataset', required=False, type=str, default="FRONT", help="dataset path")
    parser.add_argument('--exp', default='../released_full_model', help='experiment name')
    parser.add_argument('--epoch', type=str, default='100', help='saved epoch')
    parser.add_argument('--room_type', default='livingroom', help='Room type (determines class list)')
    parser.add_argument('--with_CLIP', default=True, type=bool_flag, help='Model uses CLIP')
    parser.add_argument('--save_3d', default=True, type=bool_flag, help='Save .obj and .glb files')
    parser.add_argument('--render_type', type=str, default='echoscene')
    parser.add_argument('--gen_shape', default=False, type=bool_flag, help='infer diffusion')
    args = parser.parse_args()

    argsJson = os.path.join(args.exp, 'args.json')
    assert os.path.exists(argsJson), f'Could not find args.json at {argsJson}'
    with open(argsJson) as j:
        modelArgs = json.load(j)

    modelArgs['store_path'] = os.path.join(args.exp, "vis_3dssg", args.epoch)
    os.makedirs(modelArgs['store_path'], exist_ok=True)
    
    # 1. Manually Load Vocab
    vocab = {}
    classes = {}
    classes_r = {}
    with open(os.path.join(args.dataset, f'classes_{args.room_type}.txt'), "r") as f:
        for i, line in enumerate(f):
            classes[line.strip('\n')] = i
    classes_r = {v: k for k, v in classes.items()}

    relationships = {}
    relationships_r = {}
    with open(os.path.join(args.dataset, 'relationships.txt'), "r") as f:
        for i, line in enumerate(f):
            relationships[line.strip('\n')] = i
    relationships_r = {v: k for k, v in relationships.items()}

    vocab['object_name_to_idx'] = classes
    vocab['object_idx_to_name'] = classes_r
    vocab['pred_name_to_idx'] = relationships
    vocab['pred_idx_to_name'] = relationships_r

    # 2. Load Model
    print("Loading Model...")
    diff_opt = modelArgs['diff_yaml']
    diff_cfg = OmegaConf.load(diff_opt)
    # the model expects a train_stats_file for normalization
    diff_cfg.layout_branch.diffusion_kwargs.train_stats_file = os.path.join(args.dataset, f'boxes_centered_stats_{args.room_type}_trainval.txt')
    diff_cfg.layout_branch.denoiser_kwargs.using_clip = modelArgs['with_CLIP']
    
    model = SGDiff(type=modelArgs['network_type'], diff_opt=diff_cfg, vocab=vocab, 
                   replace_latent=modelArgs.get('replace_latent', None),
                   with_changes=modelArgs.get('with_changes', None), 
                   residual=modelArgs['residual'], gconv_pooling=modelArgs['pooling'], 
                   clip=modelArgs['with_CLIP'], with_angles=modelArgs['with_angles'], 
                   separated=modelArgs['separated'])
    model.diff.optimizer_ini()
    model.load_networks(exp=args.exp, epoch=args.epoch, restart_optim=True, load_shape_branch=args.gen_shape)
    model = model.cuda()
    model.eval()

    # 3. Load 3DSSG JSON & Build Tensors
    print(f"Loading 3DSSG data from {args.json_path}...")
    with open(args.json_path, 'r') as f:
        scene_data = json.load(f)
        
    scan_id = scene_data.get('scene_id', 'adapted_scene')
    
    # We must add a 'floor' node as node 0 (if the model expects it, typically SG-FRONT scenes always have a floor)
    # Actually, let's just add the nodes given in the json. If floor is missing, we'll add it.
    nodes = scene_data['nodes']
    edges = scene_data['edges']

    # Find if there is a floor
    has_floor = any(n['class_label'] == 'floor' for n in nodes)
    floor_id = "floor_node"
    if not has_floor:
        nodes.append({"id": floor_id, "class_label": "floor"})
        
    # Map original string IDs to integer indices (0 to N-1)
    id_to_idx = {n['id']: i for i, n in enumerate(nodes)}
    
    dec_objs_list = []
    for n in nodes:
        lbl = n['class_label']
        if lbl not in classes:
            print(f"Warning: class {lbl} not in vocab. Falling back to floor.")
            lbl = "floor"
        dec_objs_list.append(classes[lbl])
        
    dec_triples_list = []
    for e in edges:
        subj, rel, obj = e
        if rel not in relationships:
            print(f"Warning: rel {rel} not in vocab. Skipping.")
            continue
        subj_idx = id_to_idx.get(subj)
        obj_idx = id_to_idx.get(obj)
        if subj_idx is None or obj_idx is None:
            continue
        dec_triples_list.append([subj_idx, relationships[rel], obj_idx])
        
    # We also connect all items to floor (standing on) if they don't have it.
    # Optional step, but helps stability.
    standing_rel = relationships.get('standing on', 7)
    for i in range(len(nodes)):
        if nodes[i]['class_label'] != 'floor':
            dec_triples_list.append([i, standing_rel, id_to_idx[floor_id]])

    dec_objs = torch.tensor(dec_objs_list, dtype=torch.long).cuda()
    dec_triples = torch.tensor(dec_triples_list, dtype=torch.long).cuda()

    # 4. Generate CLIP Embeddings
    encoded_dec_text_feat, encoded_dec_rel_feat = None, None
    if modelArgs['with_CLIP']:
        print("Extracting CLIP embeddings...")
        cond_model, _ = clip.load("ViT-B/32", device='cuda')
        
        node_feats = []
        for n in nodes:
            lbl = n['class_label']
            text_t = clip.tokenize(lbl).cuda()
            feat = cond_model.encode_text(text_t).float().detach()
            node_feats.append(feat)
        encoded_dec_text_feat = torch.cat(node_feats, dim=0) # [N, 512]
        
        rel_feats = []
        for t in dec_triples_list:
            rel_name = relationships_r[t[1]]
            text_t = clip.tokenize(rel_name).cuda()
            feat = cond_model.encode_text(text_t).float().detach()
            rel_feats.append(feat)
        if len(rel_feats) > 0:
            encoded_dec_rel_feat = torch.cat(rel_feats, dim=0) # [E, 512]
        else:
            encoded_dec_rel_feat = torch.zeros((0, 512)).cuda()

    # Objectness mask (ignore floor and _scene_ for bounding box metrics)
    objectness_mask = torch.ones(len(dec_objs), dtype=torch.bool).cuda()
    for i, idx in enumerate(dec_objs_list):
        label = classes_r[idx]
        if label in ['_scene_', 'floor']:
            objectness_mask[i] = False
    model.diff.current_objectness = objectness_mask

    # 5. Inference
    print("Running Inference...")
    with torch.no_grad():
        data_dict = model.sample_box_and_shape(dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, gen_shape=args.gen_shape)
        
        boxes_pred = torch.concat((data_dict['sizes'], data_dict['translations']), dim=-1)
        angles_pred = data_dict['angles']
        shapes_pred = data_dict.get('shapes', None)
        
        angles_pred = postprocess_sincos2arctan(angles_pred) / np.pi * 180
        normalized_file = os.path.join(args.dataset, f'centered_bounds_{args.room_type}_trainval.txt')
        boxes_pred_den = descale_box_params(boxes_pred, file=normalized_file)
        
    print("Inference complete.")
    
    # 6. Render / Save Results
    print("Rendering...")
    if shapes_pred is not None:
        shapes_pred = shapes_pred.cpu().detach()
        
    render_full(scan_id, dec_objs.cpu().numpy(), boxes_pred_den, angles_pred, 
                datasize='large' if modelArgs.get('large', True) else 'small',
                classes=classes_r, render_type=args.render_type, shapes_pred=shapes_pred, 
                store_img=True, render_boxes=False, visual=False, demo=False, 
                epoch=args.epoch, without_lamp=False, store_path=modelArgs['store_path'], 
                save_3d=args.save_3d)
                
    if args.save_3d:
        scene_dir = os.path.join(modelArgs['store_path'], args.render_type)
        mesh_dir = os.path.join(scene_dir, 'object_meshes', scan_id)
        scene_mesh_path = os.path.join(scene_dir, f"{scan_id}_{args.render_type}.glb")
        output_dir = os.path.join(modelArgs['store_path'], 'structured_scenes')
        
        export_path = export_structured_scene(
            output_dir=output_dir,
            scan_id=scan_id,
            cat_ids=dec_objs.cpu(),
            boxes=boxes_pred_den.cpu(),
            angles=angles_pred.cpu(),
            triples=dec_triples.cpu(),
            classes=classes_r,
            predicate_names=relationships_r,
            instance_ids=[],
            mesh_dir=mesh_dir,
            scene_mesh_path=scene_mesh_path,
            render_type=args.render_type,
            room_type=args.room_type,
            epoch=args.epoch,
            exp_path=args.exp,
            dataset_path=args.dataset,
            source_object_metadata=None,
            excluded_render_categories={'lamp'},
            layout_guidance=None
        )
        print(f"Structured scene exported to: {export_path}")
        
if __name__ == "__main__":
    evaluate_3dssg()
