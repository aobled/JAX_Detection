import jax
import jax.numpy as jnp

def compute_grid_loss(pred_grid, gt_boxes, lambda_coord=5.0, lambda_noobj=0.5):
    """
    Calcule la loss de détection style YOLO.
    
    Args:
        pred_grid: (Batch, S, S, 5) -> [Conf, x, y, w, h]
             - Conf: Probabilité qu'il y ait un objet
             - x, y: Centre relatif à la cellule (0-1)
             - w, h: Taille relative à l'image entière (0-1)
        
        gt_boxes: (Batch, MaxBoxes, 5) -> [HasObject, cx, cy, w, h]
             - HasObject: 1.0 si c'est un vrai objet, 0.0 sinon (padding)
             - cx, cy: Centre relatif à l'image entière (0-1)
             - w, h: Taille relative à l'image entière (0-1)
             
    Returns:
        Scalar loss
    """
    batch_size = pred_grid.shape[0]
    S = pred_grid.shape[1] # e.g. 14
    
    # 1. Construire la Target Grid (Batch, S, S, 5) à la volée
    # On doit mapper chaque GT box vers sa cellule (i, j)
    
    # Initialiser target grid à 0
    target_grid = jnp.zeros((batch_size, S, S, 5))
    
    # Indices de grille
    # Note: Vectoriser ça en JAX pur est tricky car scatter dynamique
    # On va utiliser une boucle scan ou une approche vectorisée masquée
    
    # Approche vectorisée naïve : Pour chaque box, on calcule sa cellule
    # gt_boxes: (B, N, 5)
    
    # On extrait les coordonnées
    has_obj = gt_boxes[:, :, 0] # (B, N)
    gx = gt_boxes[:, :, 1] * S  # Coordonnée x dans la grille (ex: 7.5)
    gy = gt_boxes[:, :, 2] * S
    gw = gt_boxes[:, :, 3]
    gh = gt_boxes[:, :, 4]
    
    gi = gx.astype(jnp.int32) # Col index (ex: 7)
    gj = gy.astype(jnp.int32) # Row index
    
    # Offset dans la cellule (ce que le modèle doit prédire)
    tx = gx - gi
    ty = gy - gj
    
    # Création des masques pour loss
    # On veut créer un masque (B, S, S) où 1 = il y a un objet
    # Et une target (B, S, S, 5) avec les valeurs idéales
    
    # Utilisation de .at[].set() de JAX pour remplir la grille
    # Attention: si plusieurs objets dans la même cellule, le dernier gagne (limitation YOLO v1)
    
    def build_grid_single_image(boxes):
        # boxes: (N, 5) -> [has, cx, cy, w, h]
        grid = jnp.zeros((S, S, 5))
        
        # Filtrer boxes actives (has == 1)
        valid_mask = boxes[:, 0] > 0.5
        
        # Calculer indices
        bx = boxes[:, 1] * S
        by = boxes[:, 2] * S
        col = bx.astype(jnp.int32)
        row = by.astype(jnp.int32)
        
        # Offsets
        off_x = bx - col
        off_y = by - row
        
        # Clip indices (sécurité)
        col = jnp.clip(col, 0, S-1)
        row = jnp.clip(row, 0, S-1)
        
        # Valeurs à mettre: [1.0, off_x, off_y, w, h]
        # On doit itérer car 'at[idx].set' ne gère pas collision dynamiquement facilement en vmap pur sans logique
        # Mais ici boxes est petite (30 max), donc une boucle scan est ok ou juste for loop unroll si N petit
        # JAX favorise vmap.
        
        # Astuce : On crée des updates et on scatter
        # Updates: (N, 5)
        vals = jnp.stack([
            jnp.ones_like(off_x), # Conf = 1
            off_x,
            off_y,
            boxes[:, 3], # w
            boxes[:, 4]  # h
        ], axis=-1)
        
        # On applique seulement pour valid boxes
        # Si invalid, on met à 0 (mais on scattera pas)
        # Hack: on met index à 0,0 si invalid et on masquera après ? 
        # Non, .at avec mask is better mais requires modern JAX
        
        # Approche simple : On update séquentiellement (lent mais sûr)
        # Ou mieux : scatter update
        indices = (row, col)
        
        # On ne garde que les valides pour le scatter
        # JAX n'aime pas boolean indexing dynamique pour scatter shapes
        # On utilise where pour annuler l'impact des invalides
        # (on écrit des 0 à la position 0,0 par exemple)
        
        safe_row = jnp.where(valid_mask, row, 0)
        safe_col = jnp.where(valid_mask, col, 0)
        
        # On set tout, mais pour les invalides on overwrite le 0,0 avec des zero, c'est sale.
        # Mieux: target_grid = grid.at[safe_row, safe_col].set(vals)
        # Mais ça écrase 0,0 avec la dernière box invalide.
        # Solution: Initialiser grid.
        
        grid = grid.at[safe_row, safe_col].set(vals)
        
        # Pour nettoyer le "dumping ground" en 0,0 si on a écrit des trucs invalides par erreur:
        # On recalculera le masque proprement.
        # Une box invalide a has=0 -> vals=[1, x, y, w, h].
        # Si on force vals à 0 quand !valid :
        safe_vals = vals * valid_mask[:, None] 
        grid = grid.at[safe_row, safe_col].set(safe_vals)
        
        return grid

    # Vectoriser sur le batch
    target_grids = jax.vmap(build_grid_single_image)(gt_boxes) # (B, S, S, 5)
    
    # Masque objet: target_grids[..., 0] == 1
    obj_mask = target_grids[..., 0:1] # (B, S, S, 1)
    noobj_mask = 1.0 - obj_mask
    
    # --- HANDLING ANCHORS (B > 1) ---
    C_pred = pred_grid.shape[-1]
    B_boxes = C_pred // 5
    
    if B_boxes == 1:
        # --- LOSS SINGLE BOX ---
        pred_conf = pred_grid[..., 0:1]
        pred_xy = pred_grid[..., 1:3]
        pred_wh = pred_grid[..., 3:5]
        
        target_conf = target_grids[..., 0:1]
        target_xy = target_grids[..., 1:3]
        target_wh = target_grids[..., 3:5]
        
        loss_xy = jnp.sum(obj_mask * (pred_xy - target_xy)**2)
        loss_wh = jnp.sum(obj_mask * (jnp.sqrt(jnp.abs(pred_wh) + 1e-6) - jnp.sqrt(target_wh + 1e-6))**2)
        loss_coord = lambda_coord * (loss_xy + loss_wh)
        
        loss_obj = jnp.sum(obj_mask * (pred_conf - target_conf)**2)
        loss_noobj = lambda_noobj * jnp.sum(noobj_mask * (pred_conf - target_conf)**2)
        
    else:
        # --- LOSS MULTIPLE BOXES (Anchors) ---
        pred_grid = pred_grid.reshape((batch_size, S, S, B_boxes, 5))
        target_grid_b = jnp.expand_dims(target_grids, axis=3) # (B, S, S, 1, 5)
        
        pred_xywh = pred_grid[..., 1:5]
        target_xywh = target_grid_b[..., 1:5]
        
        # Calculate L2 distance (Proxy for IoU) to find responsible box
        box_mse = jnp.sum((pred_xywh - target_xywh)**2, axis=-1) # (B, S, S, B_boxes)
        best_box_idx = jnp.argmin(box_mse, axis=-1) # (B, S, S)
        
        # Create masks
        resp_mask = jax.nn.one_hot(best_box_idx, B_boxes) # (B, S, S, B_boxes)
        obj_mask_b = obj_mask * resp_mask # responsible box gets 1 if object is present
        noobj_mask_b = 1.0 - obj_mask_b # non-responsible or background gets penalized
        
        pred_conf = pred_grid[..., 0]
        pred_xy = pred_grid[..., 1:3]
        pred_wh = pred_grid[..., 3:5]
        
        target_conf = target_grid_b[..., 0]
        target_xy = target_grid_b[..., 1:3]
        target_wh = target_grid_b[..., 3:5]
        
        loss_xy = jnp.sum(obj_mask_b * jnp.sum((pred_xy - target_xy)**2, axis=-1))
        loss_wh = jnp.sum(obj_mask_b * jnp.sum((jnp.sqrt(jnp.abs(pred_wh) + 1e-6) - jnp.sqrt(target_wh + 1e-6))**2, axis=-1))
        loss_coord = lambda_coord * (loss_xy + loss_wh)
        
        loss_obj = jnp.sum(obj_mask_b * (pred_conf - target_conf)**2)
        # target_conf is 1 where obj_mask is 1 (expanded). Where noobj_mask_b is 1, target is 0.
        loss_noobj = lambda_noobj * jnp.sum(noobj_mask_b * (pred_conf - 0.0)**2)

    # Total
    total_loss = (loss_coord + loss_obj + loss_noobj) / batch_size
    
    return total_loss

def compute_grid_loss_multilevel(pred_grids, gt_boxes, lambda_coord=5.0, lambda_noobj=0.5):
    """
    Calcule la loss YOLO multi-échelles (Dual-Scale 14x14 et 7x7).
    
    Args:
        pred_grids: Tuple de 2 tenseurs:
            - pred_14x14: (Batch, 14, 14, 10)  (2 anchors x 5 channels)
            - pred_7x7:   (Batch, 7, 7, 10)    (2 anchors x 5 channels)
        gt_boxes: (Batch, MaxBoxes, 5) -> [HasObject, cx, cy, w, h] (valeurs relatives 0-1)
    """
    pred_14, pred_7 = pred_grids
    batch_size = pred_14.shape[0]
    
    # Anchors empiriques (w, h) en coordonnées normalisées relatives à l'image complète
    # 2 anchors pour 14x14 (petits avions/drones), 2 anchors pour 7x7 (gros avions en gros plan)
    anchors_14 = jnp.array([[0.1, 0.1], [0.15, 0.08]]) # Carré petit, Rectangulaire moyen
    anchors_7  = jnp.array([[0.4, 0.4], [0.6, 0.3]])   # Carré grand, Rectangulaire large
    all_anchors = jnp.concatenate([anchors_14, anchors_7], axis=0) # (4, 2)
    
    def compute_iou_wh(box_wh, anchors_wh):
        # Calcule l'IoU uniquement sur la largeur et hauteur (Prior Matching)
        intersection = jnp.minimum(box_wh[0], anchors_wh[:, 0]) * jnp.minimum(box_wh[1], anchors_wh[:, 1])
        box_area = box_wh[0] * box_wh[1]
        anchors_area = anchors_wh[:, 0] * anchors_wh[:, 1]
        union = box_area + anchors_area - intersection + 1e-6
        return intersection / union
        
    def build_target_single_image(boxes):
        target_14 = jnp.zeros((14, 14, 2, 5))
        target_7 = jnp.zeros((7, 7, 2, 5))
        
        def step_fn(state, box):
            t_14, t_7 = state
            has_obj = box[0]
            cx, cy, w, h = box[1:5]
            
            # IoU avec les 4 anchors
            ious = compute_iou_wh(jnp.array([w, h]), all_anchors)
            best_idx = jnp.argmax(ious) # 0, 1, 2, ou 3
            
            # Logique pour la grille 14x14
            cx_14, cy_14 = cx * 14.0, cy * 14.0
            col_14 = jnp.clip(cx_14.astype(jnp.int32), 0, 13)
            row_14 = jnp.clip(cy_14.astype(jnp.int32), 0, 13)
            tx_14 = cx_14 - col_14
            ty_14 = cy_14 - row_14
            val_14 = jnp.array([1.0, tx_14, ty_14, w, h])
            
            # Logique pour la grille 7x7
            cx_7, cy_7 = cx * 7.0, cy * 7.0
            col_7 = jnp.clip(cx_7.astype(jnp.int32), 0, 6)
            row_7 = jnp.clip(cy_7.astype(jnp.int32), 0, 6)
            tx_7 = cx_7 - col_7
            ty_7 = cy_7 - row_7
            val_7 = jnp.array([1.0, tx_7, ty_7, w, h])
            
            # Appliquer conditionnellement
            # Fait partie de 14x14 si best_idx est 0 ou 1
            cond_14 = jnp.logical_and(has_obj > 0.5, best_idx < 2)
            anchor_idx_14 = best_idx
            new_t_14 = t_14.at[row_14, col_14, anchor_idx_14].set(val_14)
            t_14 = jnp.where(cond_14, new_t_14, t_14)
            
            # Fait partie de 7x7 si best_idx est 2 ou 3
            cond_7 = jnp.logical_and(has_obj > 0.5, best_idx >= 2)
            anchor_idx_7 = best_idx - 2
            new_t_7 = t_7.at[row_7, col_7, anchor_idx_7].set(val_7)
            t_7 = jnp.where(cond_7, new_t_7, t_7)
            
            return (t_14, t_7), None

        # JAX scannera toutes les boxes (par exemple les 30 boxes paddées de notre max)
        (final_14, final_7), _ = jax.lax.scan(step_fn, (target_14, target_7), boxes)
        return final_14, final_7

    # Vmap sur tout le batch
    target_14, target_7 = jax.vmap(build_target_single_image)(gt_boxes)
    # (Batch, 14, 14, 2, 5) et (Batch, 7, 7, 2, 5)
    
    # Reshapes des dimensions des tenseurs prédictions
    pred_14 = pred_14.reshape(batch_size, 14, 14, 2, 5)
    pred_7 = pred_7.reshape(batch_size, 7, 7, 2, 5)
    
    def compute_single_scale_loss(pred_g, tgt_g):
        obj_mask = tgt_g[..., 0:1]
        noobj_mask = 1.0 - obj_mask
        
        pred_conf = pred_g[..., 0:1]
        pred_xy = pred_g[..., 1:3]
        pred_wh = pred_g[..., 3:5]
        
        tgt_conf = tgt_g[..., 0:1]
        tgt_xy = tgt_g[..., 1:3]
        tgt_wh = tgt_g[..., 3:5]
        
        loss_xy = jnp.sum(obj_mask * (pred_xy - tgt_xy)**2)
        loss_wh = jnp.sum(obj_mask * (jnp.sqrt(jnp.abs(pred_wh) + 1e-6) - jnp.sqrt(tgt_wh + 1e-6))**2)
        
        loss_obj = jnp.sum(obj_mask * (pred_conf - tgt_conf)**2)
        loss_noobj = lambda_noobj * jnp.sum(noobj_mask * (pred_conf - 0.0)**2)
        
        return lambda_coord * (loss_xy + loss_wh) + loss_obj + loss_noobj

    # Calculer séparément puis additionner
    loss_14 = compute_single_scale_loss(pred_14, target_14)
    loss_7 = compute_single_scale_loss(pred_7, target_7)
    
    total_loss = (loss_14 + loss_7) / batch_size
    return total_loss


def compute_v7_loss(pred_grids, gt_boxes, lambda_coord=5.0, lambda_noobj=0.5):
    """
    Calcule la loss Anchor-Free Tri-Scale pour V7.
    
    Args:
        pred_grids: Tuple de 3 tenseurs:
            - pred_28: (Batch, 28, 28, 5)
            - pred_14: (Batch, 14, 14, 5)
            - pred_7:  (Batch, 7, 7, 5)
        gt_boxes: (Batch, MaxBoxes, 5) -> [HasObject, cx, cy, w, h] (valeurs relatives 0-1)
    """
    pred_28, pred_14, pred_7 = pred_grids
    batch_size = pred_28.shape[0]
    
    # Stratégie d'assignation basée sur l'aire de la bounding box (anchor-free FPN)
    # Aire relative = w * h (0-1)
    # P2 (28x28) : pour les petits objets (area < 0.05)
    # P3 (14x14) : pour les objets moyens (0.05 <= area < 0.20)
    # P4 (7x7)   : pour les grands objets (area >= 0.20)
    
    def build_target_single_image(boxes):
        target_28 = jnp.zeros((28, 28, 5))
        target_14 = jnp.zeros((14, 14, 5))
        target_7 = jnp.zeros((7, 7, 5))
        
        def step_fn(state, box):
            t_28, t_14, t_7 = state
            has_obj = box[0]
            cx, cy, w, h = box[1:5]
            
            area = w * h
            
            # --- Logique P2 (28x28) ---
            cx_28, cy_28 = cx * 28.0, cy * 28.0
            col_28 = jnp.clip(cx_28.astype(jnp.int32), 0, 27)
            row_28 = jnp.clip(cy_28.astype(jnp.int32), 0, 27)
            tx_28 = cx_28 - col_28
            ty_28 = cy_28 - row_28
            val_28 = jnp.array([1.0, tx_28, ty_28, w, h])
            
            # --- Logique P3 (14x14) ---
            cx_14, cy_14 = cx * 14.0, cy * 14.0
            col_14 = jnp.clip(cx_14.astype(jnp.int32), 0, 13)
            row_14 = jnp.clip(cy_14.astype(jnp.int32), 0, 13)
            tx_14 = cx_14 - col_14
            ty_14 = cy_14 - row_14
            val_14 = jnp.array([1.0, tx_14, ty_14, w, h])
            
            # --- Logique P4 (7x7) ---
            cx_7, cy_7 = cx * 7.0, cy * 7.0
            col_7 = jnp.clip(cx_7.astype(jnp.int32), 0, 6)
            row_7 = jnp.clip(cy_7.astype(jnp.int32), 0, 6)
            tx_7 = cx_7 - col_7
            ty_7 = cy_7 - row_7
            val_7 = jnp.array([1.0, tx_7, ty_7, w, h])
            
            # Assignation FPN
            cond_28 = jnp.logical_and(has_obj > 0.5, area < 0.05)
            cond_14 = jnp.logical_and(has_obj > 0.5, jnp.logical_and(area >= 0.05, area < 0.20))
            cond_7  = jnp.logical_and(has_obj > 0.5, area >= 0.20)
            
            new_t_28 = t_28.at[row_28, col_28].set(val_28)
            t_28 = jnp.where(cond_28, new_t_28, t_28)
            
            new_t_14 = t_14.at[row_14, col_14].set(val_14)
            t_14 = jnp.where(cond_14, new_t_14, t_14)
            
            new_t_7 = t_7.at[row_7, col_7].set(val_7)
            t_7 = jnp.where(cond_7, new_t_7, t_7)
            
            return (t_28, t_14, t_7), None

        (final_28, final_14, final_7), _ = jax.lax.scan(step_fn, (target_28, target_14, target_7), boxes)
        return final_28, final_14, final_7

    target_28, target_14, target_7 = jax.vmap(build_target_single_image)(gt_boxes)
    
    def compute_single_scale_loss(pred_g, tgt_g):
        obj_mask = tgt_g[..., 0:1]
        noobj_mask = 1.0 - obj_mask
        
        pred_conf = pred_g[..., 0:1]
        pred_xy = pred_g[..., 1:3]
        pred_wh = pred_g[..., 3:5]
        
        tgt_conf = tgt_g[..., 0:1]
        tgt_xy = tgt_g[..., 1:3]
        tgt_wh = tgt_g[..., 3:5]
        
        loss_xy = jnp.sum(obj_mask * (pred_xy - tgt_xy)**2)
        loss_wh = jnp.sum(obj_mask * (jnp.sqrt(jnp.abs(pred_wh) + 1e-6) - jnp.sqrt(tgt_wh + 1e-6))**2)
        
        loss_obj = jnp.sum(obj_mask * (pred_conf - tgt_conf)**2)
        loss_noobj = lambda_noobj * jnp.sum(noobj_mask * (pred_conf - 0.0)**2)
        
        return lambda_coord * (loss_xy + loss_wh) + loss_obj + loss_noobj

    loss_28 = compute_single_scale_loss(pred_28, target_28)
    loss_14 = compute_single_scale_loss(pred_14, target_14)
    loss_7 = compute_single_scale_loss(pred_7, target_7)
    
    total_loss = (loss_28 + loss_14 + loss_7) / batch_size
    return total_loss
