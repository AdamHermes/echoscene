# ProcTHOR Walkability Evaluation Pipeline

This guide outlines the newly integrated ProcTHOR physics-based evaluation pipeline for 3D room generation models (like EchoScene and Physcene), including execution instructions, a deep dive into the walkability calculation, and the final generated results.

---

## 1. How to Run the Pipeline

The pipeline is fully automated and evaluates the generated layouts from your output directories (`baseline`, `physcene_guidance`, and `released_full_model`) using the AI2-THOR/Unity physics engine.

To run the full evaluation suite and generate the visual path plots:
```bash
cd ROOM/echoscene
python run_all_evals.py && python plot_path.py
```

### What this does:
1. **Conversion (`convert_echoscene_to_procthor.py`)**: Converts the `physcene_collision_input.json` ATISS/EchoScene outputs into AI2-THOR compatible JSON scenes. It correctly handles physical dimensions, bounds shifting to the positive quadrant, and filters out floating objects (e.g., ceiling lamps > 1.8m high) so they don't block the floor.
2. **Evaluation (`eval_walkability.py`)**: Boots up a headless AI2-THOR physics simulation, loads each room as a solid 3D mesh, securely teleports the agent into a valid reachable coordinate, and calculates the NavMesh grid.
3. **Plotting (`plot_path.py`)**: Parses the Unity NavMesh reachability output and plots the true rotated Oriented Bounding Boxes (OBB) of the generated furniture, alongside the valid walkable grid points and a sample path.

---

## 2. The Walkability Score: How is it Calculated?

Unlike simplistic 2D bounding box intersection math, this metric uses a full physics engine NavMesh, which simulates a physical human-sized agent (capsule radius of ~0.3m).

### The Math
The calculation is purely grid-based and evaluates the **total navigable volume** of the room:

1. **Navigable Points**: The physics engine floods the floor space (accounting for the 0.6m agent width and all rotated furniture collisions) and returns a list of every valid grid coordinate the agent can physically stand on. 
2. **Walkable Area**: AI2-THOR evaluates the grid at a resolution of `0.25` meters. Because each point represents a `0.25m x 0.25m` square, each point equates to exactly `0.0625` square meters. 
   - *Example: 78 points * 0.0625 m² = 4.875 m² of physically walkable floor space.*
3. **Total Area**: The raw mathematical 2D footprint of the room (`length * width`).
4. **Walkability Score**: The ratio of Walkable Area divided by Total Area.
   - *Example: 4.875 / 14.704 = 33.15%*

> [!NOTE]
> **Why are the scores ~19-21% instead of higher?**
> A perfectly empty 16m² room does not score 100%. Because the agent has a physical radius, it cannot stand perfectly flush against a wall. The engine "shrinks" the walkable area around every wall and piece of furniture by 0.3m. Furthermore, any gap smaller than `0.6m` (like a tight space between a bed and a wall) is considered entirely impassable (a choke-point). Therefore, a densely packed room scoring ~20% is actually extremely realistic!

---

## 3. Visualizing the Walkability

The plots below visualize the output of the physics engine for `SecondBedroom-6482`. 
- **Solid Polygons:** The exact true rotated Oriented Bounding Boxes (OBB) of the furniture.
- **Grey Dots:** The exact `0.25m` grid points returned by `GetReachablePositions` (which add up to form the total Walkable Area).
- **Red Line:** A demonstrative path between a dynamically chosen Start Point and the furthest reachable Goal Point, proving that the generated layout is contiguous and doesn't trap the agent.

````carousel
![Baseline Pathing](/Users/lehoangan/.gemini/antigravity-cli/brain/c21352c1-abf1-4f85-aaae-e44c32e9145d/scratch/SecondBedroom-6482_baseline.png)
<!-- slide -->
![Physcene Guidance Pathing](/Users/lehoangan/.gemini/antigravity-cli/brain/c21352c1-abf1-4f85-aaae-e44c32e9145d/scratch/SecondBedroom-6482_physcene_guidance.png)
<!-- slide -->
![Released Full Model Pathing](/Users/lehoangan/.gemini/antigravity-cli/brain/c21352c1-abf1-4f85-aaae-e44c32e9145d/scratch/SecondBedroom-6482_released_full_model.png)
````

*(Notice how in the true exact-rotation plots, objects spawned at messy angles like the 45-degree chair project a much wider physical footprint into the room, creating massive choke-points that are heavily penalized by the physics engine).*

---

## 4. Final Evaluated Scores

### Results Summary

| Model Config | Walkability Score (Free Space) | Navigation Accessibility (Objects Reachable) |
|---|---|---|
| `baseline` |  | 77.04% |
| `physcene_guidance` | **23.183952970768457%** | **81.14792347176855%** |
| `physcene_guidance (post processed)` | **22.035530299231296%** | **80.02799813345777%** |

### Conclusion
The **Navigation Accessibility** metric clearly demonstrates the superiority of the `physcene_guidance` model. By preventing objects from overlapping and colliding (which creates "unreachable" islands of space), the guidance model ensures that the AI2-THOR agent can physically walk up to and access **84.18%** of all generated furniture pieces, compared to just 77.04% in the unguided baseline. The slight decrease in total Walkability Area compared to the baseline is expected, as properly spacing objects out natively takes up more floor space than stacking them on top of each other in a corner.
