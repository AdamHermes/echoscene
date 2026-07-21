from __future__ import print_function

import open3d as o3d # open3d needs to be imported before other packages!
import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import gc

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from model.SGDiff import SGDiff
from dataset.threedfront_dataset import ThreedFrontDatasetSceneGraph
from helpers.util import bool_flag, preprocess_angle2sincos, batch_torch_destandardize_box_params, descale_box_params, postprocess_sincos2arctan, sample_points
from helpers.metrics_3dfront import validate_constrains, validate_constrains_changes, estimate_angular_std
from helpers.visualize_scene import render_full, render_box
from helpers.structured_scene_export import export_structured_scene
from omegaconf import OmegaConf
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, type=str, default="/media/ymxlzgy/Data/Dataset/FRONT", help="dataset path")
parser.add_argument('--with_CLIP', type=bool_flag, default=True, help="Load Feats directly instead of points.")

parser.add_argument('--manipulate', default=True, type=bool_flag)
parser.add_argument('--exp', default='../released_full_model', help='experiment name')
parser.add_argument('--epoch', type=str, default='100', help='saved epoch')
parser.add_argument('--render_type', type=str, default='txt2shape', help='retrieval, txt2shape, onlybox, echoscene')
parser.add_argument('--gen_shape', default=False, type=bool_flag, help='infer diffusion')
parser.add_argument('--visualize', default=False, type=bool_flag)
parser.add_argument('--export_3d', default=False, type=bool_flag, help='Export the generated shapes and boxes in json files for future use')
parser.add_argument('--room_type', default='all', help='all, bedroom, livingroom, diningroom, library')
parser.add_argument('--max_samples', type=int, default=None, help='Limit evaluation to the first N samples')
parser.add_argument('--start_idx', type=int, default=0, help='Start evaluation from this sample index')
parser.add_argument('--save_3d', default=True, type=bool_flag, help='Save .obj and .glb files')
parser.add_argument('--default_exp', default='../released_full_model', help='default exp load arguments')
parser.add_argument('--debug', default=False, type=bool_flag, help='Print debug bbox info')
args = parser.parse_args()

room_type = ['all', 'bedroom', 'livingroom', 'diningroom', 'library']


def reseed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize(vertices, scale=1):
    xmin, xmax = np.amin(vertices[:, 0]), np.amax(vertices[:, 0])
    ymin, ymax = np.amin(vertices[:, 1]), np.amax(vertices[:, 1])
    zmin, zmax = np.amin(vertices[:, 2]), np.amax(vertices[:, 2])

    vertices[:, 0] += -xmin - (xmax - xmin) * 0.5
    vertices[:, 1] += -ymin - (ymax - ymin) * 0.5
    vertices[:, 2] += -zmin - (zmax - zmin) * 0.5

    scalars = np.max(vertices, axis=0)
    scale = scale

    vertices = vertices / scalars * scale
    return vertices


def export_echoscene_sidecar(modelArgs, test_dataset, data, dec_objs, dec_triples, boxes_pred_den, angles_pred,
                             classes, epoch=None, render_type='echoscene'):
    scan_id = data['scan_id'][0]
    instance_ids = data['instance_id'][0] if len(data['instance_id']) > 0 else []
    scene_dir = os.path.join(modelArgs['store_path'], render_type)
    mesh_dir = os.path.join(scene_dir, 'object_meshes', scan_id)
    scene_mesh_path = os.path.join(scene_dir, "{0}_{1}.glb".format(scan_id, render_type))
    output_dir = os.path.join(modelArgs['store_path'], 'structured_scenes')

    source_object_metadata = None
    if hasattr(test_dataset, 'tight_boxes_json') and scan_id in test_dataset.tight_boxes_json:
        source_object_metadata = test_dataset.tight_boxes_json[scan_id]

    layout_guidance = None
    if hasattr(test_dataset, 'eval_type'):
        layout_guidance = {'eval_type': test_dataset.eval_type}

    export_path = export_structured_scene(
        output_dir=output_dir,
        scan_id=scan_id,
        cat_ids=dec_objs.detach().cpu(),
        boxes=boxes_pred_den.detach().cpu(),
        angles=angles_pred.detach().cpu(),
        triples=dec_triples.detach().cpu(),
        classes=classes,
        predicate_names=test_dataset.vocab['pred_idx_to_name'],
        instance_ids=instance_ids,
        mesh_dir=mesh_dir,
        scene_mesh_path=scene_mesh_path,
        render_type=render_type,
        room_type=getattr(test_dataset, 'room_type', None),
        epoch=epoch,
        exp_path=args.exp,
        dataset_path=args.dataset,
        source_object_metadata=source_object_metadata,
        excluded_render_categories={'lamp'},
        layout_guidance=layout_guidance,
    )
    print("structured scene exported:", export_path)
    return export_path

def validate_constrains_loop_w_changes(modelArgs, testdataset, model, normalized_file=None, bin_angles=False, cat2objs=None, datasize='large', gen_shape=False):

    test_dataloader_changes = torch.utils.data.DataLoader(
        testdataset,
        batch_size=1,
        collate_fn=testdataset.collate_fn,
        shuffle=False,
        num_workers=0)
    vocab = testdataset.vocab
    obj_classes = testdataset.classes_r
    pred_classes = vocab['pred_idx_to_name']

    accuracy = {}
    accuracy_unchanged = {}
    accuracy_in_orig_graph = {}

    for k in ['left', 'right', 'front', 'behind', 'smaller', 'bigger', 'shorter', 'taller', 'standing on', 'close by', 'symmetrical to', 'total']:
        accuracy_in_orig_graph[k] = []
        accuracy_unchanged[k] = []
        accuracy[k] = []

    for i, data in enumerate(test_dataloader_changes, 0):
        print(data['scan_id'][0])

        try:
            enc_objs, enc_triples, enc_objs_to_scene, enc_triples_to_scene = data['encoder']['objs'], \
                                                                                              data['encoder']['tripltes'], \
                                                                                              data['encoder']['obj_to_scene'], \
                                                                                              data['encoder']['triple_to_scene']

            dec_objs, dec_triples, dec_tight_boxes, dec_objs_to_scene, dec_triples_to_scene = data['decoder']['objs'], \
                                                                                              data['decoder']['tripltes'], \
                                                                                              data['decoder']['boxes'], \
                                                                                              data['decoder']['obj_to_scene'], \
                                                                                              data['decoder']['triple_to_scene']
            dec_sdfs = None
            if modelArgs['with_SDF']:
                dec_sdfs = data['decoder']['sdfs']

            missing_nodes = data['missing_nodes']
            manipulated_subs = data['manipulated_subs']
            manipulated_objs = data['manipulated_objs']
            manipulated_preds = data['manipulated_preds']

        except Exception as e:
            print("Exception: skipping scene", e)
            continue

        enc_objs, enc_triples = enc_objs.cuda(), enc_triples.cuda()
        dec_objs, dec_triples, dec_tight_boxes = dec_objs.cuda(), dec_triples.cuda(), dec_tight_boxes.cuda()
        encoded_enc_rel_feat, encoded_enc_text_feat, encoded_dec_text_feat, encoded_dec_rel_feat = None, None, None, None
        if modelArgs['with_CLIP']:
            encoded_enc_text_feat, encoded_enc_rel_feat = data['encoder']['text_feats'].cuda(), data['encoder']['rel_feats'].cuda()
            encoded_dec_text_feat, encoded_dec_rel_feat = data['decoder']['text_feats'].cuda(), data['decoder']['rel_feats'].cuda()

        all_pred_boxes = []
        all_pred_angles = []

        class_idx = dec_objs.cpu().numpy().astype(int)
        objectness_mask = torch.ones(len(dec_objs), dtype=torch.bool, device=dec_objs.device)
        for i, idx in enumerate(class_idx):
            label = obj_classes[int(idx)].strip('\n')
            if label in ['_scene_', 'floor']:
                objectness_mask[i] = False
        model.diff.current_objectness = objectness_mask
        model.diff.current_gt_boxes = dec_tight_boxes

        with torch.no_grad():
            original = 0
            if original:
                # original graph
                print("***original graph***")
                original_data_dict = model.sample_box_and_shape(enc_objs, enc_triples, encoded_enc_text_feat, encoded_enc_rel_feat,
                                                       gen_shape=gen_shape)
                original_boxes_pred, original_angles_pred = torch.concat((original_data_dict['sizes'], original_data_dict['translations']), dim=-1), original_data_dict['angles']
                original_shapes_pred = None
                try:
                    original_shapes_pred = original_data_dict['shapes']
                except:
                    print('no shape, only run layout branch.')
                original_angles_pred = postprocess_sincos2arctan(original_angles_pred) / np.pi * 180
                original_boxes_pred = descale_box_params(original_boxes_pred, file=normalized_file)  # min, max

            # manipulated graph
            print("***manipulated graph***")
            if len(manipulated_subs) and len(manipulated_objs):
                manipulated_nodes = manipulated_subs + manipulated_objs
                print('previous:' , obj_classes[int(enc_objs[manipulated_subs[0]].item())], pred_classes[int(manipulated_preds[0].item())], obj_classes[int(enc_objs[manipulated_objs[0]].item())])
                keep, data_dict = model.sample_boxes_and_shape_with_changes(enc_objs, enc_triples, encoded_enc_text_feat,
                                                                            encoded_enc_rel_feat, dec_objs, dec_triples,
                                                                            encoded_dec_text_feat, encoded_dec_rel_feat,
                                                                            manipulated_nodes, gen_shape=gen_shape)
            else:
                keep, data_dict = model.sample_boxes_and_shape_with_additions(enc_objs, enc_triples, encoded_enc_text_feat,
                                                                              encoded_enc_rel_feat, dec_objs, dec_triples,
                                                                              encoded_dec_text_feat, encoded_dec_rel_feat,
                                                                              missing_nodes, gen_shape=gen_shape)

            boxes_pred, angles_pred = torch.concat((data_dict['sizes'], data_dict['translations']), dim=-1), data_dict['angles']
            shapes_pred = None
            try:
                shapes_pred = data_dict['shapes']
            except:
                print('no shape, only run layout branch.')

            if modelArgs['bin_angle']:
                angles_pred = -180 + (torch.argmax(angles_pred, dim=1, keepdim=True) + 1)* 15.0
                boxes_pred_den = batch_torch_destandardize_box_params(boxes_pred, file=normalized_file)
            else:
                angles_pred = postprocess_sincos2arctan(angles_pred) / np.pi * 180
                boxes_pred_den = descale_box_params(boxes_pred, file=normalized_file)

            if args.visualize:
                print("rendering", [obj_classes[i.item()].strip('\n') for i in dec_objs])
                if model.type_ == 'echoscene':
                    if original:
                        if original_shapes_pred is not None:
                            original_shapes_pred = original_shapes_pred.cpu().detach()
                        render_full(data['scan_id'], enc_objs.detach().cpu().numpy(), original_boxes_pred, original_angles_pred,
                                    datasize=datasize,
                                    classes=obj_classes, render_type=args.render_type, shapes_pred=original_shapes_pred,
                                    store_img=True,
                                    render_boxes=False, visual=True, demo=True, without_lamp=False,
                                    store_path=modelArgs['store_path']+"_before",save_3d=args.save_3d)

                    if shapes_pred is not None:
                        shapes_pred = shapes_pred.cpu().detach()
                    render_full(data['scan_id'], dec_objs.detach().cpu().numpy(), boxes_pred_den, angles_pred,
                                datasize=datasize,
                                classes=obj_classes, render_type=args.render_type, shapes_pred=shapes_pred, store_img=True,
                                render_boxes=False, visual=True, demo=True, without_lamp=False,
                                store_path=modelArgs['store_path']+"_after",save_3d=args.save_3d)
                else:
                    raise NotImplementedError

        bp_box, bp_angle = [], []
        for i in range(len(keep)):
            if keep[i] == 0:
                bp_box.append(boxes_pred_den[i:i+1].cpu().detach())
                bp_angle.append(angles_pred[i:i+1].cpu().detach())
            else:
                dec_tight_boxes[i:i+1,:6] = descale_box_params(dec_tight_boxes[i:i+1,:6], file=normalized_file)
                bp_box.append(dec_tight_boxes[i:i+1,:6].cpu().detach())
                angle = dec_tight_boxes[i:i+1, 6:7] / np.pi * 180
                bp_angle.append(angle.cpu().detach())

        all_pred_boxes.append(boxes_pred_den.cpu().detach())
        all_pred_angles.append(angles_pred.cpu().detach())

        accuracy = validate_constrains_changes(dec_triples, boxes_pred_den, angles_pred, keep, model.vocab, accuracy)
        accuracy_unchanged = validate_constrains(dec_triples, boxes_pred_den, angles_pred, keep, model.vocab, accuracy_unchanged)

    keys = list(accuracy.keys())
    file_path_for_output = os.path.join(modelArgs['store_path'], f'{testdataset.eval_type}_accuracy_analysis.txt')
    with open(file_path_for_output, 'w') as file:
        for dic, typ in [(accuracy, "changed nodes"), (accuracy_unchanged, 'unchanged nodes')]:
            lr_mean = np.mean([np.mean(dic[keys[0]]), np.mean(dic[keys[1]])])
            fb_mean = np.mean([np.mean(dic[keys[2]]), np.mean(dic[keys[3]])])
            bism_mean = np.mean([np.mean(dic[keys[4]]), np.mean(dic[keys[5]])])
            tash_mean = np.mean([np.mean(dic[keys[6]]), np.mean(dic[keys[7]])])
            stand_mean = np.mean(dic[keys[8]])
            close_mean = np.mean(dic[keys[9]])
            symm_mean = np.mean(dic[keys[10]])
            total_mean = np.mean(dic[keys[11]])
            means_of_mean = np.mean([lr_mean, fb_mean, bism_mean, tash_mean, stand_mean, close_mean, symm_mean])
            print('{} & L/R: {:.2f} & F/B: {:.2f} & Bi/Sm: {:.2f} & Ta/Sh: {:.2f} & Stand: {:.2f} & Close: {:.2f} & Symm: {:.2f}. Total: &{:.2f}'.format(typ, lr_mean,
                                        fb_mean, bism_mean, tash_mean, stand_mean, close_mean, symm_mean, total_mean))
            print('means of mean: {:.2f}'.format(means_of_mean))
            file.write(
                '{} & L/R: {:.2f} & F/B: {:.2f} & Bi/Sm: {:.2f} & Ta/Sh: {:.2f} & Stand: {:.2f} & Close: {:.2f} & Symm: {:.2f}. Total: &{:.2f}\n'.format(
                    typ, lr_mean, fb_mean, bism_mean, tash_mean, stand_mean, close_mean, symm_mean, total_mean))
            file.write('means of mean: {:.2f}\n\n'.format(means_of_mean))


def validate_constrains_loop(modelArgs, test_dataset, model, epoch=None, normalized_file=None, cat2objs=None, datasize='large', gen_shape=False):

    test_dataloader_no_changes = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=test_dataset.collate_fn,
        shuffle=False,
        num_workers=0)

    vocab = test_dataset.vocab

    accuracy = {}
    for k in ['left', 'right', 'front', 'behind', 'smaller', 'bigger', 'shorter', 'taller', 'standing on', 'close by', 'symmetrical to', 'total']:
        accuracy[k] = []
    physcene_export = {
        "class_labels": [], "translations": [], "sizes": [],
        "angles": [], "objfeats_32": [], "objectness": [], "scene_ids": []
    }
    for i, data in enumerate(test_dataloader_no_changes, 0):
        print(data['scan_id'])

        try:
            dec_objs, dec_triples, dec_tight_boxes = data['decoder']['objs'], data['decoder']['tripltes'], data['decoder']['boxes']
            instances = data['instance_id'][0]
            scan = data['scan_id'][0]
        except Exception as e:
            print(e)
            continue

        dec_objs, dec_triples = dec_objs.cuda(), dec_triples.cuda()
        encoded_dec_text_feat, encoded_dec_rel_feat = None, None
        if modelArgs['with_CLIP']:
            encoded_dec_text_feat, encoded_dec_rel_feat = data['decoder']['text_feats'].cuda(), data['decoder']['rel_feats'].cuda()

        all_pred_boxes = []
        all_pred_angles = []

        class_idx = dec_objs.cpu().numpy().astype(int)
        objectness_mask = torch.ones(len(dec_objs), dtype=torch.bool, device=dec_objs.device)
        for i, idx in enumerate(class_idx):
            label = test_dataset.classes_r[int(idx)].strip('\n')
            if label in ['_scene_', 'floor']:
                objectness_mask[i] = False
        model.diff.current_objectness = objectness_mask
        model.diff.current_gt_boxes = dec_tight_boxes.cuda()

        with torch.no_grad():

            data_dict = model.sample_box_and_shape(dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, gen_shape=gen_shape)

            boxes_pred, angles_pred = torch.concat((data_dict['sizes'],data_dict['translations']),dim=-1), data_dict['angles']
            shapes_pred = None
            try:
                shapes_pred = data_dict['shapes']
            except:
                print('no shape, only run layout branch.')
            if modelArgs['bin_angle']:
                angles_pred = -180 + (torch.argmax(angles_pred, dim=1, keepdim=True) + 1)* 15.0
                boxes_pred_den = batch_torch_destandardize_box_params(boxes_pred, file=normalized_file)
            else:
                angles_pred = postprocess_sincos2arctan(angles_pred) / np.pi * 180
                boxes_pred_den = descale_box_params(boxes_pred, file=normalized_file)

        if args.debug:
            # ── BBOX DEBUG PRINT ──────────────────────────────────────────────
            debug_log_path = os.path.join(modelArgs['store_path'], 'debug_bbox.txt')
            os.makedirs(modelArgs['store_path'], exist_ok=True)
            
            with open(debug_log_path, 'a' if i > 0 else 'w') as dbg_file:
                def dprint(msg):
                    print(msg)
                    dbg_file.write(msg + '\n')

                classes_sorted = test_dataset.classes_r
                obj_ids = dec_objs.detach().cpu().numpy()
                boxes_np  = boxes_pred_den.detach().cpu().numpy()   # (N, 6): [l, h, w, x, y, z]
                angles_np = angles_pred.detach().cpu().numpy()      # (N, 1): degrees

                dprint(f"\n{'='*60}")
                dprint(f"SCENE: {data['scan_id'][0]}  |  {len(obj_ids)} objects")
                dprint(f"{'='*60}")
                dprint(f"{'Obj':<20} {'l':>6} {'h':>6} {'w':>6}  {'x':>7} {'y':>7} {'z':>7}  {'angle':>7}")
                dprint(f"{'-'*70}")
                for n in range(len(obj_ids)):
                    name = classes_sorted[int(obj_ids[n])].strip('\n')
                    l, h, w = boxes_np[n, 0], boxes_np[n, 1], boxes_np[n, 2]
                    x, y, z = boxes_np[n, 3], boxes_np[n, 4], boxes_np[n, 5]
                    a = float(angles_np[n])
                    dprint(f"{name:<20} {l:>6.3f} {h:>6.3f} {w:>6.3f}  {x:>7.3f} {y:>7.3f} {z:>7.3f}  {a:>7.2f}°")

                # pairwise bounding-box IoU to flag colliders immediately
                dprint(f"\n  Pairwise overlap check (axis-aligned bbox):")
                any_collision = False
                for ni in range(len(obj_ids)):
                    for nj in range(ni + 1, len(obj_ids)):
                        bi = boxes_np[ni]   # [l,h,w,x,y,z]
                        bj = boxes_np[nj]
                        # compute axis-aligned extents
                        min_i = bi[3:6] - bi[0:3] / 2.0
                        max_i = bi[3:6] + bi[0:3] / 2.0
                        min_j = bj[3:6] - bj[0:3] / 2.0
                        max_j = bj[3:6] + bj[0:3] / 2.0
                        overlap = np.all(max_i > min_j) and np.all(max_j > min_i)
                        if overlap:
                            any_collision = True
                            name_i = classes_sorted[int(obj_ids[ni])].strip('\n')
                            name_j = classes_sorted[int(obj_ids[nj])].strip('\n')
                            # compute overlap depth on each axis as a rough severity measure
                            depth = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
                            dprint(f"  !! OVERLAP: {name_i} <-> {name_j}  "
                                  f"depth=[{depth[0]:.3f}, {depth[1]:.3f}, {depth[2]:.3f}]")
                if not any_collision:
                    dprint("  (no axis-aligned overlaps)")
                dprint(f"{'='*60}\n")
            # ── END BBOX DEBUG ────────────────────────────────────────────────

        entry = build_physcene_json_entry(
            dec_objs        = dec_objs,
            boxes_pred_den  = boxes_pred_den,
            angles_pred     = angles_pred,     # degrees, shape (N,1)
            obj_classes     = test_dataset.classes_r,
            scan_id         = data['scan_id'][0],
        )
        physcene_export["class_labels"].append(entry["class_labels"])
        physcene_export["translations"].append(entry["translations"])
        physcene_export["sizes"].append(entry["sizes"])
        physcene_export["angles"].append(entry["angles"])
        physcene_export["objfeats_32"].append(entry["objfeats_32"])
        physcene_export["objectness"].append(entry["objectness"])
        physcene_export["scene_ids"].append(entry["scene_id"])
        if args.visualize:
            classes = test_dataset.classes_r
            print("rendering", [classes[i.item()].strip('\n') for i in dec_objs])
            if model.type_ == 'echolayout':
                render_box(data['scan_id'], dec_objs.detach().cpu().numpy(), boxes_pred_den, angles_pred, datasize=datasize,
                classes=classes, render_type=args.render_type, store_img=False, render_boxes=False, visual=False, demo=False, without_lamp=False, store_path=modelArgs['store_path'],save_3d=args.save_3d)
            elif model.type_ == 'echoscene':
                if shapes_pred is not None:
                    shapes_pred = shapes_pred.cpu().detach()
                render_full(data['scan_id'], dec_objs.detach().cpu().numpy(), boxes_pred_den, angles_pred, datasize=datasize,
                classes=classes, render_type=args.render_type, shapes_pred=shapes_pred, store_img=True, render_boxes=False, visual=False, demo=False,epoch=epoch, without_lamp=False, store_path=modelArgs['store_path'],save_3d=args.save_3d)
                if args.export_3d:
                    if not args.save_3d:
                        print("Skipping structured scene export because --save_3d is False.")
                    else:
                        export_echoscene_sidecar(modelArgs, test_dataset, data, dec_objs, dec_triples,
                                                 boxes_pred_den, angles_pred, classes, epoch=epoch,
                                                 render_type=args.render_type)
            else:
                raise NotImplementedError

        all_pred_boxes.append(boxes_pred_den.cpu().detach())
        all_pred_angles.append(angles_pred.cpu().detach())
        accuracy = validate_constrains(dec_triples, boxes_pred_den, angles_pred, None, model.vocab, accuracy)
        
        export_path = os.path.join(
            modelArgs['store_path'], 'physcene_collision_input.json'
        )
        os.makedirs(modelArgs['store_path'], exist_ok=True)
        with open(export_path, 'w') as f:
            json.dump(physcene_export, f)
            
        # Free memory at the end of the loop to prevent VRAM spikes across scenes
        del data_dict, boxes_pred, angles_pred, boxes_pred_den
        if shapes_pred is not None:
            del shapes_pred
        gc.collect()
        torch.cuda.empty_cache()

    keys = list(accuracy.keys())

    file_path_for_output = os.path.join(modelArgs['store_path'], f'{test_dataset.eval_type}_accuracy_analysis.txt')
    with open(file_path_for_output, 'w') as file:
        for dic, typ in [(accuracy, "acc")]:
            lr_mean = np.mean([np.mean(dic[keys[0]]), np.mean(dic[keys[1]])])
            fb_mean = np.mean([np.mean(dic[keys[2]]), np.mean(dic[keys[3]])])
            bism_mean = np.mean([np.mean(dic[keys[4]]), np.mean(dic[keys[5]])])
            tash_mean = np.mean([np.mean(dic[keys[6]]), np.mean(dic[keys[7]])])
            stand_mean = np.mean(dic[keys[8]])
            close_mean = np.mean(dic[keys[9]])
            symm_mean = np.mean(dic[keys[10]])
            total_mean = np.mean(dic[keys[11]])
            means_of_mean = np.mean([lr_mean, fb_mean, bism_mean, tash_mean, stand_mean, close_mean, symm_mean])
            print(
                '{} & L/R: {:.2f} & F/B: {:.2f} & Bi/Sm: {:.2f} & Ta/Sh: {:.2f} & Stand: {:.2f} & Close: {:.2f} & Symm: {:.2f}. Total: &{:.2f}'.format(
                    typ, lr_mean,
                    fb_mean, bism_mean, tash_mean, stand_mean, close_mean, symm_mean, total_mean))
            print('means of mean: {:.2f}'.format(means_of_mean))
            file.write(
                '{} & L/R: {:.2f} & F/B: {:.2f} & Bi/Sm: {:.2f} & Ta/Sh: {:.2f} & Stand: {:.2f} & Close: {:.2f} & Symm: {:.2f}. Total: &{:.2f}\n'.format(
                    typ, lr_mean, fb_mean, bism_mean, tash_mean, stand_mean, close_mean, symm_mean, total_mean))
            file.write('means of mean: {:.2f}\n\n'.format(means_of_mean))
    export_path = os.path.join(
        modelArgs['store_path'], 'physcene_collision_input.json'
    )
    os.makedirs(modelArgs['store_path'], exist_ok=True)
    with open(export_path, 'w') as f:
        json.dump(physcene_export, f)
    print(f"PhyScene input saved to: {export_path}")

def build_physcene_json_entry(
    dec_objs,        # (N_obj,) int tensor — class indices
    boxes_pred_den,  # (N_obj, 6) float tensor — [l,h,w,x,y,z] denormalized
    angles_pred,     # (N_obj, 1) float tensor — degrees (from postprocess_sincos2arctan)
    obj_classes,
    scan_id,
):
    """
    Convert EchoScene per-scene output to PhyScene JSON format.
    Returns dict with numpy arrays ready for JSON serialization.
    """

    n_classes   = len(obj_classes)
    N           = dec_objs.shape[0]

    # ── Sizes: EchoScene boxes_pred_den = [l, h, w, x, y, z]
    sizes_np = boxes_pred_den[:, 0:3].cpu().numpy()        # (N, 3) in meters

    # ── Translations
    trans_np = boxes_pred_den[:, 3:6].cpu().numpy()        # (N, 3) in meters

    # ── Angles: EchoScene gives degrees → convert to radians
    angles_deg = angles_pred.cpu().numpy()                  # (N, 1) degrees
    angles_rad = angles_deg / 180.0 * np.pi                # (N, 1) radians

    # ── Class labels: integer → one-hot (N, n_classes+1)
    #    last column = empty/padding class (PhyScene convention)
    class_idx  = dec_objs.cpu().numpy().astype(int)        # (N,)
    one_hot    = np.zeros((N, n_classes + 1), dtype=np.float32)
    for i, idx in enumerate(class_idx):
        label = obj_classes[int(idx)].strip('\n')
        if label in ['_scene_', 'floor']:
            one_hot[i, n_classes] = 1.0    # mark as empty/padding
        else:
            one_hot[i, idx] = 1.0

    # ── Objectness: 0 for _scene_/floor, 1 for real objects
    objectness = np.zeros((N, 1), dtype=np.float32)
    for i, idx in enumerate(class_idx):
        label = obj_classes[int(idx)].strip('\n')
        if label not in ['_scene_', 'floor']:
            objectness[i, 0] = 1.0

    # ── objfeats_32: zeros (PhyScene uses this for mesh retrieval only)
    objfeats = np.zeros((N, 32), dtype=np.float32)

    return {
        "class_labels": one_hot.tolist(),    # (N, n_classes+1)
        "translations": trans_np.tolist(),   # (N, 3)
        "sizes":        sizes_np.tolist(),   # (N, 3)
        "angles":       angles_rad.tolist(), # (N, 1)
        "objfeats_32":  objfeats.tolist(),   # (N, 32)
        "objectness":   objectness.tolist(), # (N, 1)
        "scene_id":     scan_id,
    }
def evaluate():
    reseed(48)

    argsJson = os.path.join(args.exp, 'args.json')
    assert os.path.exists(argsJson), 'Could not find args.json for experiment {}'.format(args.exp)
    with open(argsJson) as j:
        modelArgs = json.load(j)
    normalized_file = os.path.join(args.dataset, 'centered_bounds_{}_trainval.txt').format(modelArgs['room_type'])
    # test_dataset_rels_changes = ThreedFrontDatasetSceneGraph(
        # root=args.dataset,
        # split='val_scans',
        # use_scene_rels=modelArgs['use_scene_rels'],
        # with_changes=True,
        # eval=True,
        # eval_type='relationship',
        # with_CLIP=modelArgs['with_CLIP'],
        # use_SDF=modelArgs['with_SDF'],
        # large=modelArgs['large'],
        # room_type=args.room_type,
        # recompute_clip=False)

    # test_dataset_addition_changes = ThreedFrontDatasetSceneGraph(
    #     root=args.dataset,
    #     split='val_scans',
    #     use_scene_rels=modelArgs['use_scene_rels'],
    #     with_changes=True,
    #     eval=True,
    #     eval_type='addition',
    #     with_CLIP=modelArgs['with_CLIP'],
    #     use_SDF=modelArgs['with_SDF'],
    #     large=modelArgs['large'],
    #     room_type=args.room_type)

    test_dataset_no_changes = ThreedFrontDatasetSceneGraph(
        root=args.dataset,
        split='val_scans',
        use_scene_rels=modelArgs['use_scene_rels'],
        with_changes=False,
        eval=True,
        eval_type='none',
        with_CLIP=modelArgs['with_CLIP'],
        use_SDF=modelArgs['with_SDF'],
        large=modelArgs['large'],
        room_type=args.room_type)

    # apply start_idx and max_samples slicing only if specified
    if args.max_samples is not None:
        test_dataset_no_changes.scans = test_dataset_no_changes.scans[args.start_idx:args.start_idx + args.max_samples]
        #test_dataset_rels_changes.scans = test_dataset_rels_changes.scans[args.start_idx:args.start_idx + args.max_samples]
        #test_dataset_addition_changes.scans = test_dataset_addition_changes.scans[args.start_idx:args.start_idx + args.max_samples]

    modeltype_ = modelArgs['network_type']
    modelArgs['store_path'] = os.path.join(args.exp, "vis", args.epoch)
    replacelatent_ = modelArgs['replace_latent'] if 'replace_latent' in modelArgs else None
    with_changes_ = modelArgs['with_changes'] if 'with_changes' in modelArgs else None

    diff_opt = modelArgs['diff_yaml']
    diff_cfg = OmegaConf.load(diff_opt)
    diff_cfg.layout_branch.diffusion_kwargs.train_stats_file = test_dataset_no_changes.box_normalized_stats
    diff_cfg.layout_branch.denoiser_kwargs.using_clip = modelArgs['with_CLIP']
    model = SGDiff(type=modeltype_, diff_opt=diff_cfg, vocab=test_dataset_no_changes.vocab, replace_latent=replacelatent_,
                with_changes=with_changes_, residual=modelArgs['residual'], gconv_pooling=modelArgs['pooling'], clip=modelArgs['with_CLIP'],
                with_angles=modelArgs['with_angles'], separated=modelArgs['separated'])
    model.diff.optimizer_ini()
    model.load_networks(exp=args.exp, epoch=args.epoch, restart_optim=True, load_shape_branch=args.gen_shape)
    if torch.cuda.is_available():
        model = model.cuda()

    model = model.eval()
    cat2objs = None

    print('Evaluating {} sample(s)'.format(len(test_dataset_no_changes)))

    print('\nEditing Mode - Additions')
    reseed(47)
    # validate_constrains_loop_w_changes(modelArgs, test_dataset_addition_changes, model, normalized_file=normalized_file, bin_angles=modelArgs['bin_angle'], cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

    reseed(47)
    print('\nEditing Mode - Relationship changes')
    # validate_constrains_loop_w_changes(modelArgs, test_dataset_rels_changes, model,  normalized_file=normalized_file, bin_angles=modelArgs['bin_angle'], cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

    reseed(47)
    print('\nGeneration Mode')
    validate_constrains_loop(modelArgs, test_dataset_no_changes, model, epoch=args.epoch, normalized_file=normalized_file, cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

if __name__ == "__main__":
    print(torch.__version__)
    evaluate()
