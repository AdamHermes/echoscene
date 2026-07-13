# ProcTHOR Evaluation Scripts

This directory contains documentation and instructions for evaluating EchoScene layouts using AI2-THOR / ProcTHOR.

## Files and Scripts
All scripts are located in the main `echoscene` directory and run from there.

1. **`convert_echoscene_to_procthor.py`**: Reads `final.json` from the generated outputs, maps the bounding boxes to ProcTHOR room boundaries, and converts the furniture into solid walls (with ceiling lids) to guarantee exact collision sizes in Unity.
2. **`eval_walkability.py`**: Computes the ratio of empty floor space (NavMesh) relative to total room area. Note that overlapping objects artificially inflate this score.
3. **`eval_navigation.py`**: Computes **Accessibility** by spawning an AI2-THOR agent and testing if every single generated object is reachable within 0.75m. This is a 2D equivalent of ProcTHOR's native `GetInteractablePoses` metric.
4. **`plot_path.py`**: Visualizes the AI2-THOR native `GetShortestPathToPoint` routing around the generated layouts.
5. **`run_convert.sh`**: Helper shell script to convert `baseline`, `physcene_guidance`, and `released_full_model` simultaneously.
6. **`run_all_evals.py`**: Python orchestrator to run walkability, accessibility, and plotting across all datasets.

## How to run the pipeline
To evaluate a new generation run (e.g. `output/physcene_guidance`):

1. **Convert to ProcTHOR JSON**:
   ```bash
   python convert_echoscene_to_procthor.py --bbox_path ./output/physcene_guidance/vis/2050/final.json --out_dir ./output/physcene_guidance/vis/2050/procthor_scenes
   ```
2. **Run Metrics**:
   ```bash
   python eval_walkability.py --scenes_dir ./output/physcene_guidance/vis/2050/procthor_scenes
   python eval_navigation.py --scenes_dir ./output/physcene_guidance/vis/2050/procthor_scenes
   ```

The results are saved directly in the `procthor_scenes` folder as `walkability_results.json` and `navigation_results.json`.
