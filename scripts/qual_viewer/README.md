# Qualitative-figure picker

Browser tool for choosing scenes and views for the paper's qualitative figures. It shows the same test scene under
each method side by side in 3D, with one camera shared by all panels.

- `server.py` runs on the machine with the generated meshes (lhanubu, SPINNINGHDD). It needs numpy and trimesh.
- `config_lhanubu.json` sets the runs and the panels. The default panels are the paper's four variants:
  - EchoScene;
  - + Collision Resolver;
  - + Physical Guidance;
  - + SteerScene.
- Scenes are ranked so that those where the other panels have collisions, out-of-room objects or objects moved by the resolver, and ours are clean, come first.
  - Collisions are mesh-level, taken from `eval_mesh_collision.py --out` files in `mesh_eval/`, or computed if missing.
  - Out-of-room uses rotated footprints, as in `eval_ob.py`.

## Run

```bash
./run.sh            # first start builds qual_cache/index.json
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

Picks are saved on the server in `qual_cache/picks.json`. Export options:

- **Picks table:** rows are methods and columns are the picked scenes, each with its saved view (TokenCompose-style).
- **This scene:** a–d grid in the style of Fig. 1.
- **One PNG per panel:** separate images for this scene.
