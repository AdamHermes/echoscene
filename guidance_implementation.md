# Physical Guidance Implementation in EchoScene

This document explains the implementation of physical guidance during the reverse diffusion process, detailing how collision loss, room outer loss, and walkable loss are calculated and integrated into the model based on the changes between commits `5779882bcb3073e5531d0039b6a10c31244f4781` and `fdc78e87fa88be6de744674c28fa86d1edd78f66`.

## 1. Overview of Guidance Implementation

Physical guidance is applied during the inference stage of the denoising diffusion probabilistic model (DDPM) to enforce physical constraints such as avoiding object collisions and keeping objects within the room boundaries. It acts as an energy-based classifier-free guidance mechanism.

The core implementation resides in `model/networks/diffusion_layout/diffusion_ddpm.py`, specifically within the `_apply_inference_guidance` and `p_sample_sg` methods:
- At each scheduled denoising timestep, the neural network predicts the unnoised layout (`pred_xstart`). From this prediction, the analytical `model_mean` is computed. The `model_variance` is deterministically derived from the fixed diffusion noise schedule (e.g. `fixedlarge` or `fixedsmall`).
- The predicted layout parameters are denormalized from the standard normal space to their physical dimensions (sizes, translations, angles) using `_denormalize_box_params`.
- A composite physical loss (`total_guidance_loss`) is computed using the denormalized boxes.
- The gradient of this total loss with respect to the continuous representation `pred_xstart` is computed via backpropagation (`torch.autograd.grad`).
- The computed gradient is clipped (if configured) and multiplied by the `model_variance`.
- The `model_mean` is shifted in the opposite direction of the gradient scaled by a guidance `strength`: 
  ```python
  variance_preconditioned_grad = model_variance * guidance_grad
  guided_mean = model_mean - strength * variance_preconditioned_grad
  ```
- The denoised sample for the next step is then drawn using this `guided_mean` instead of the original predicted mean.

## 2. Collision Loss Calculation

The collision loss penalizes intersecting objects within the layout. It is implemented in the `_compute_collision_guidance_loss` function in `diffusion_ddpm.py`.

**Calculation Steps:**
1. **Scene and Object Masking:** The batch is separated into individual scenes, and background architecture objects (e.g., floor, `_scene_`) are masked out using an `objectness` tensor so that collisions with them are ignored.
2. **3D Intersection-over-Union (IoU):** The 3D IoU between all pairs of objects is computed using `cal_iou_3d` (which projects oriented 3D boxes).
3. **Physical Penetration (AABB Approximation):** Since accurately calculating the intersection volume of two rotated 3D boxes is highly complex and slow to compute during training, the `_compute_pairwise_penetration` function uses a faster, axis-aligned bounding box (AABB) approximation:
   - **Step A (Rotated Footprint):** First, it finds a new, unrotated "safety box" that fully encloses the rotated object. Since objects only rotate on the floor plane (yaw angle), the vertical height ($Y$) stays the same. For the $X$ (width) and $Z$ (depth) dimensions, it uses trigonometry to project the corners of the rotated box onto the fixed axes. The new width becomes `width * cos(angle) + depth * sin(angle)`, and the new depth becomes `width * sin(angle) + depth * cos(angle)`.
   - **Step B (Half-Sizes):** These new, expanded dimensions are divided by 2 to get the "half-sizes"—the distance from the object's center to the edges of its bounding box.
   - **Step C (Center Distances):** The absolute distance between the center points of every pair of objects is calculated along the X, Y, and Z axes.
   - **Step D (Overlap Calculation):** For two boxes to intersect, the distance between their centers must be *less* than the sum of their half-sizes. The formula `(half_size_A + half_size_B) - distance_between_centers` computes this.
     - If the result is **negative**, there is empty space between the objects on that axis.
     - If the result is **positive**, the objects are overlapping by that exact amount on that axis.
   - **Step E (ReLU and Volume):** A Rectified Linear Unit (ReLU) function is applied to clamp any negative values to zero. The penetration volume is then calculated by multiplying the overlapping amounts across all three axes ($X \times Y \times Z$).
4. **Loss Aggregation:** For object pairs exceeding specified `iou_threshold` and `penetration_threshold`, the loss calculates the mean active IoU and active penetration depth. The final collision loss is a weighted sum (`iou_weight * active_pair_iou.mean() + penetration_weight * active_pair_penetration.mean()`).

## 3. Room Outer Loss Calculation

The room outer loss ensures that generated objects do not penetrate the architectural boundaries of the room (the walls). It is implemented in `compute_room_outer_loss` within `physical_guidance.py`.

**Calculation Steps:**
1. **Boundary Identification:** The function identifies the floor object by inspecting objects marked as background (`~objectness`). It selects the largest background object by its X-Z area and sets the room bounds (`max_bound_x`, `min_bound_x`, `max_bound_z`, `min_bound_z`) to the extents of this floor object. If no floor object is available, it defaults to a hardcoded $6m \times 6m$ boundary (`[-3.0, 3.0]`).
2. **Object Extents:** For each generated object, it computes the bounding box corners relative to the centers:
   ```python
   max_corners = centers_obj + half_sizes_obj
   min_corners = centers_obj - half_sizes_obj
   ```
3. **L1 Penalty:** An L1 distance penalty is applied wherever an object's boundary exceeds the identified room bounds using `torch.relu`.
   ```python
   loss_x_max = torch.relu(max_corners[:, 0] - max_bound_x).sum()
   loss_x_min = torch.relu(min_bound_x - min_corners[:, 0]).sum()
   # Repeated for Z axis
   ```
4. The penalties across the X and Z axes are summed to form the `room_outer_loss`.

## 4. Walkable Loss Calculation

The walkable loss (or reachability guidance) ensures that an agent (e.g., a human or robot) can freely traverse the generated room without being completely blocked by furniture. This loss is implemented in the `compute_walkable_loss` and `calc_loss_on_path` functions within `physical_guidance.py`.

**Calculation Steps:**
1. **Floor Plan Mapping:** The room's floor plan (vertices and faces) is projected onto a 2D grid image (representing the X-Z plane). The architectural walls are drawn to mask out the room's boundaries.
2. **Object Rasterization:** All valid generated objects (filtered to exclude high-hanging objects like ceiling lamps using the `robot_hight_real` parameter) are scaled and drawn onto this 2D map as oriented bounding boxes.
3. **Walkable Region Identification:** Using OpenCV's connected components analysis (`cv2.connectedComponentsWithStats`), the system identifies distinct "empty" regions in the room that are separated by furniture. 
4. **Shortest Path Search:** If the room is divided into two or more isolated walkable areas, a shortest-path algorithm (A* search with a heuristic distance) is used to find an optimal traversal path connecting the centers of the two largest isolated regions, effectively forcing a path *through* the blocking furniture.
5. **Path Collision Penalty:** The 2D path coordinates are converted back into a sequence of 3D bounding boxes (representing the physical volume of the navigating robot along the path). An Intersection-over-Union (IoU) overlap is computed between these "robot path boxes" and the generated furniture. The total IoU overlap across the path constitutes the `walkable_loss`, which penalizes furniture configurations that obstruct critical pathways.

**Legacy Fallback (The Old Walkable Loss):**
Prior to the full `PhyScene` port, or in cases where the architectural `floor_plan` is missing, the code falls back to the original, much simpler "center-penalty" implementation:
- It simply takes the $X$ and $Z$ center coordinates of every object.
- It calculates the squared distance of each object from the absolute center of the room `(0, 0)`.
- It applies an exponential penalty (`torch.exp(-distance_squared / 0.5)`).
This legacy approach creates an invisible, high-penalty "hill" in the dead center of the room to blindly push furniture toward the edges, but it cannot account for complex room shapes or door placements like the new A* method does.

## 5. Model Integration

These losses are tightly integrated into the reverse diffusion sampling loop:
- In `model/networks/diffusion_layout/echo2layout.py`, the variables `floor_plan`, `room_outer_box`, and `objectness` are accepted in the `set_input` process and passed through the `sample` function to the diffusion instance.
- Inside `GaussianDiffusion.p_sample_loop_sg` (`diffusion_ddpm.py`), the inference loop invokes `p_sample_sg` for each timestep.
- Within `_apply_inference_guidance`, the losses are aggregated:
  ```python
  total_guidance_loss = collision_loss * 10 + room_outer_loss * 10 + walkable_loss
  ```
- The gradients generated from `total_guidance_loss` adjust the sampling mean.
- Additionally, `p_sample_loop_sg` freezes the non-object background items (like floors) at their ground-truth configuration by overwriting their generated noise properties with the noisy version of their known ground truth:
  ```python
  if getattr(self, 'objectness', None) is not None and getattr(self, 'gt_boxes', None) is not None:
      mask = ~self.objectness.bool()
      x_t[mask] = noised_gt[mask]
  ```
This enables the layout model to guide the placement iteratively in real physical space without explicit programmatic bounding boxes, relying entirely on the differentiability of the physical loss implementations.
