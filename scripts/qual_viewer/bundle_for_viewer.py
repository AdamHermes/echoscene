"""Copy what the qualitative viewer needs for InstructScene and PhyScene (placements, floors, and only the 3D-FUTURE
meshes those placements use) to the SPINNINGHDD on lhanubu. Runs on the GPU server.

Layout on the HDD: qual_bundle/<backbone>/runs/<run>/{placements*.json, result*.json.floor.json},
qual_bundle/<backbone>/meshes/<jid>/raw_model.obj.
usage: python3 bundle_for_viewer.py
"""
import json, os, subprocess, tempfile

S = os.path.expanduser('~/LHAN/SteerScene')
DEST = 'lhan@lhan.duckdns.org:/media/lhan/SPINNINGHDD/qual_bundle'
SSH = 'ssh -o BatchMode=yes -p 2222'
IS = S + '/InstructScene/out/%s_sg2scdiffusion_objfeat/generated_scenes/epoch_01999/off_test_%s_pl'
RUNS = {
    'instructscene': {'%s_%s' % (room, run): IS % (room, run)
                      for room in ('bedroom', 'diningroom', 'livingroom') for run in ('none', 'tuned')},
    'physcene': {r: S + '/results/physcene/livingroom/' + r for r in ('physcene_coll', 'raw', 'steerscene_p_w3_r10')},
}
IS_MESHES = S + '/InstructScene/dataset/3D-FRONT/3D-FUTURE-model'
FILES = ('placements.json', 'placements_resolved.json', 'result.json.floor.json', 'result_resolved.json.floor.json')

for bb, runs in RUNS.items():
    meshes = {}
    for name, d in runs.items():
        for f in FILES[:2]:
            for sc in json.load(open(os.path.join(d, f))):
                for o in sc['objects']:
                    meshes[o['jid']] = o.get('mesh') or os.path.join(IS_MESHES, o['jid'], 'raw_model.obj')
        subprocess.run(['rsync', '-a', '-e', SSH, '--mkpath'] + [os.path.join(d, f) for f in FILES if os.path.exists(os.path.join(d, f))]
                       + ['%s/%s/runs/%s/' % (DEST, bb, name)], check=True)
    # rsync the meshes by jid: one temp tree of symlinks, copied with -L
    with tempfile.TemporaryDirectory() as tmp:
        for jid, p in meshes.items():
            os.makedirs(os.path.join(tmp, jid))
            os.symlink(p, os.path.join(tmp, jid, 'raw_model.obj'))
        subprocess.run(['rsync', '-aL', '-e', SSH, '--mkpath', tmp + '/', '%s/%s/meshes/' % (DEST, bb)], check=True)
    print(bb, len(runs), 'runs,', len(meshes), 'meshes', flush=True)
print('BUNDLE_DONE', flush=True)
