# Qualitative-figure picker

Browser tool for choosing scenes and views for the paper's qualitative figures. It shows the same test scene under
each method side by side in 3D, with one camera shared by all panels.

- `server.py` runs on the machine with the scenes (lhanubu, SPINNINGHDD). It needs numpy and trimesh.
- `config_lhanubu.json` lists one entry per backbone. Each entry has:
  - its runs (raw layout, and after the collision resolver);
  - the panels;
  - presets: which panels to show, e.g. all four for the Fig. 1 2×2, or only base and SteerScene for the big table.

Scenes are ranked so that those where the other panels have collisions, out-of-room objects or objects moved by the resolver, and ours are clean, come first.

| Backbone | What is shown | Collisions used for ranking | Data |
|---|---|---|---|
| EchoScene | generated shapes, fitted into boxes as in the FID renders | mesh-level (`mesh_eval/`) | `current_works_v2/{baseline_server,final_r6_cr30rot10_eg2_rel100}` |
| InstructScene | retrieved 3D-FUTURE furniture, from `placements.json` | rotated-box overlap | runs `none` and `tuned`, 3 rooms (`qual_bundle/instructscene`) |
| PhyScene | retrieved furniture, placed as PhyScene's evaluator does (`dump_placements.py`) | rotated-box overlap | `physcene_coll` (its own guidance) and `steerscene_p_w3_r10` (`qual_bundle/physcene`) |
| DiffuScene | boxes only, base and resolver | rotated-box overlap | the co-author's 70 scenes (`qual_bundle/diffuscene`). No guided run is available yet. |

The InstructScene and PhyScene placements, floors and the meshes they use are copied from the GPU server by
`bundle_for_viewer.py` (both run on the GPU server; copies of `revision/backbones/...`).

## Run

```bash
./run.sh            # builds qual_cache/index_<backbone>.json, then pre-builds every scene file in the background
./run.sh --rebuild  # after changing runs; ./run.sh stop to stop
```

Open http://localhost:8765. From another machine, forward the port first:

```bash
ssh -N -L 8765:localhost:8765 lhan@lhan.duckdns.org -p 22
```

## Use

| | |
|---|---|
| ← → / ↑ ↓ | previous / next scene, ±10 |
| drag | rotate all panels together |
| pinch, ctrl+scroll | zoom |
| two-finger scroll | pan (or zoom, set in the sidebar) |
| right-drag, shift+drag | pan |
| P | pick the scene and store the view; on a picked scene, update the view |
| 1–9 | switch that panel between raw and after the resolver |
| H, F, G, R | highlights, floor, 1 row / 2×2, reset view |

The **Backbone** and **Panels** menus are in the header. Picks are saved on the server, one file per backbone (`qual_cache/picks_<backbone>.json`). Export options:

- **Picks table:** rows are the shown panels and columns are the picked scenes, each with its saved view.
- **This scene:** grid of the shown panels.
- **One PNG per panel:** separate images for this scene.
