"""100% Standalone, Dependency-Free Script for BBox IOU guidance."""

import torch

def pure_pytorch_iou_3d(box1, box2):
    """
    Pure PyTorch differentiable 3D Axis-Aligned Bounding Box IoU.
    Assumes box format: [..., 7] -> (x, y, z, w, h, l, angle)
    """
    # Extract centers and sizes
    centers1, sizes1 = box1[..., :3], box1[..., 3:6]
    centers2, sizes2 = box2[..., :3], box2[..., 3:6]
    
    # Calculate min and max coordinates for AABB
    min1 = centers1 - sizes1 / 2.0
    max1 = centers1 + sizes1 / 2.0
    
    min2 = centers2 - sizes2 / 2.0
    max2 = centers2 + sizes2 / 2.0
    
    # Calculate intersection bounds
    inter_min = torch.max(min1, min2)
    inter_max = torch.min(max1, max2)
    
    # Calculate intersection volume (clamp at 0 for no overlap)
    inter_sizes = torch.clamp(inter_max - inter_min, min=0.0)
    inter_vol = inter_sizes[..., 0] * inter_sizes[..., 1] * inter_sizes[..., 2]
    
    # Calculate union volume
    vol1 = sizes1[..., 0] * sizes1[..., 1] * sizes1[..., 2]
    vol2 = sizes2[..., 0] * sizes2[..., 1] * sizes2[..., 2]
    union_vol = vol1 + vol2 - inter_vol
    
    # Calculate IoU
    iou = inter_vol / torch.clamp(union_vol, min=1e-6)
    return iou


# Notice: No @OPTIMIZER.register() and no base class inheritance!
class CollisionOptimizer:

    def __init__(self, cfg, device='cpu') -> None:
        self.device = device
        self.scale = cfg.scale
        self.scale_type = cfg.scale_type
        
        self.collision = cfg.collision
        self.collision_weight = cfg.collision_weight
        self.clip_grad_by_value = cfg.clip_grad_by_value
        self.collision_type = cfg.collision_type
        self.guidance = cfg.guidance

        self.d_class = 0
        self.d_bbox = 0

    def collision_loss(self, bbox, objectness, class_labels):
        loss_collision = 0.0
         
        if self.collision_type == "bbox_IOU":
            for j in range(len(bbox)):
                
                bbox_cur = bbox[j:j+1,:,:]
                objectness_cur = objectness[j:j+1,:,:]

                # Only evaluate active objects
                valid_mask = objectness_cur[0, :, 0] > 0
                if not valid_mask.any():
                    continue

                bbox_cur = bbox_cur[:, valid_mask, :]
                bbox_cur_cnt = bbox_cur.shape[1] 
                
                if bbox_cur_cnt < 2:
                    continue # No collision possible with < 2 objects

                for i in range(bbox_cur_cnt):    
                    bbox_target = bbox_cur[:,i,:]  # 1,7
                    bbox_target = torch.tile(bbox_target[:,None,:], [1,bbox_cur_cnt,1])   # 1, N, 7
                    
                    # Use our pure PyTorch IoU instead of the compiled cuda_op
                    loss_iter = pure_pytorch_iou_3d(bbox_cur, bbox_target)
                    
                    valid_pair = torch.ones_like(loss_iter).int()
                    valid_pair[:,i] = 0 # Ignore self-collision
                    
                    loss_iter = loss_iter * valid_pair  
                    loss_collision += loss_iter.sum() / bbox_cur_cnt / len(bbox)
                    
            loss_collision = loss_collision * 0.075
        else:
            raise ValueError(f"Unsupported collision_type: {self.collision_type}")
        
        return loss_collision

    def optimize(self, x: torch.Tensor, data, objectness=None) -> torch.Tensor:
        """ Compute gradient for optimizer constraint """
        
        translations = x[:, :, :3]
        sizes = x[:, :, 3:6] * 2  
        sizes = torch.relu(sizes) 
        angles = x[:, :, 6:7]
        class_labels = x[:, :, 7:]
        
        bbox = torch.cat([translations, sizes, angles], dim=-1)

        # permute (x,z,y,w,l,h,alpha)->(x,y,z,w,h,l,alpha)
        bbox_reordered = bbox.clone()
        bbox_reordered[:,:,1] = translations[:,:,2]
        bbox_reordered[:,:,2] = translations[:,:,1]
        bbox_reordered[:,:,4] = sizes[:,:,2]  # z
        bbox_reordered[:,:,5] = sizes[:,:,1]  # y

        # Collision loss
        loss_collision = 0.0
        if self.guidance.collision:
            loss_collision = self.collision_loss(bbox_reordered, objectness, class_labels)

        loss = loss_collision * self.guidance.weight_coll 
        return (-1.0) * loss

    def gradient(self, x: torch.Tensor, data, variance: torch.Tensor, room_outer_box=None, doors=None, floor_plan=None, floor_plan_centroid=None, objectness=None) -> torch.Tensor:
        """ Compute gradient for optimizer constraint """
        
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)  

            with torch.autograd.set_detect_anomaly(True):
                obj = self.optimize(x_in, data, objectness=objectness)
                
                if obj == 0 or not torch.is_tensor(obj):
                    return None
                    
                grad = torch.autograd.grad(obj, x_in)[0]
                
                # only keep gradient of translation x, y
                grad[:,:,1] = 0  # z
                grad[:,:,3:6] = 0  # sizes
                grad[:,:,8:] = 0  # classes
                
                # clip gradient by value
                grad = torch.clip(grad, **self.clip_grad_by_value)
            
            if self.scale_type == 'normal':
                grad = self.scale * grad * variance
            elif self.scale_type == 'div_var':
                grad = self.scale * grad
            else:
                raise Exception('Unsupported scale type!')

            return grad