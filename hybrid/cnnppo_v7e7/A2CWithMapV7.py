'''
Change from V6: Changed scout and guard channels in internal map representation: instead of decrementing by 0.01, divide by 2

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 4, 1, kernel_size=1)
        
    def forward(self, x):
        # Compute attention weights
        attention = F.relu(self.conv1(x))
        attention = self.conv2(attention)
        attention_weights = torch.sigmoid(attention)
        
        # Apply attention weights to the input feature map
        return x * attention_weights.expand_as(x), attention_weights


class A2CWithMap(nn.Module):
    def __init__(self, map_size=(16,16), map_channels=11, num_actions=5):
        super(A2CWithMap, self).__init__()
        self.map_height, self.map_width = map_size # e.g., (16, 16)
        self.map_channels = map_channels
        self.num_actions = num_actions

        self.AGENT_POS_CHANNEL = 0
        self.UNKNOWN_CHANNEL = 1
        self.RECON_CHANNEL = 2
        self.MISSION_CHANNEL = 3
        self.RIGHT_WALL_CHANNEL = 4
        self.BOTTOM_WALL_CHANNEL = 5
        self.LEFT_WALL_CHANNEL = 6
        self.TOP_WALL_CHANNEL = 7
        self.VISITED_CHANNEL = 8
        self.SCOUT_CHANNEL = 9
        self.GUARD_CHANNEL = 10

        # --- Deeper Map Processing CNN Layers ---
        self.map_cnn1 = nn.Conv2d(self.map_channels, 32, kernel_size=3, stride=1, padding=1)
        
        # Add attention after first conv layer
        self.attention1 = SpatialAttention(32)
        
        self.map_cnn2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: 64x8x8
        
        # Add attention after second conv+pool
        self.attention2 = SpatialAttention(64)
        
        self.map_cnn3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.map_cnn4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: 128x4x4
        
        # Add attention after final conv layers
        self.attention3 = SpatialAttention(128)

        # Calculate flattened size after the deeper CNN with attention
        dummy_map_input = torch.zeros(1, self.map_channels, self.map_height, self.map_width)
        dummy_map_output = F.relu(self.map_cnn1(dummy_map_input))
        dummy_map_output, _ = self.attention1(dummy_map_output)
        dummy_map_output = F.relu(self.map_cnn2(dummy_map_output))
        dummy_map_output = self.maxpool1(dummy_map_output)
        dummy_map_output, _ = self.attention2(dummy_map_output)
        dummy_map_output = F.relu(self.map_cnn3(dummy_map_output))
        dummy_map_output = F.relu(self.map_cnn4(dummy_map_output))
        dummy_map_output = self.maxpool2(dummy_map_output)
        dummy_map_output, _ = self.attention3(dummy_map_output)
        dummy_map_output = dummy_map_output.view(1, -1)
        
        self.flattened_map_features_size = dummy_map_output.size(1)
        self.layernorm_cnn = nn.LayerNorm(self.flattened_map_features_size)

        # V5 new: Embedding for scout binary input
        self.scout_embedding1 = nn.Embedding(num_embeddings=2, embedding_dim=8)

        self.combined_features_size = self.flattened_map_features_size + 9
        # Actor (Policy) Head
        self.fc_actor1 = nn.Linear(self.combined_features_size, 256)
        self.layernorm_actor1 = nn.LayerNorm(256)
        self.fc_actor2 = nn.Linear(256, 256)
        self.layernorm_actor2 = nn.LayerNorm(256)
        self.fc_actor3 = nn.Linear(256, num_actions)

        # Critic (Value) Head
        self.fc_critic1 = nn.Linear(self.combined_features_size, 256)
        self.layernorm_critic1 = nn.LayerNorm(256)
        self.fc_critic2 = nn.Linear(256, 1)

    def batched_viewcone_to_map_coords(self, direction: torch.LongTensor, location: torch.LongTensor):
        B = direction.shape[0]
        H, W = 7, 5
        device = direction.device
        # 1. Relative grid
        center_r, center_c = 2, 2
        rel_rows = (torch.arange(H, device=device).unsqueeze(1) - center_r).expand(H, W)
        rel_cols = (torch.arange(W, device=device).unsqueeze(0) - center_c).expand(H, W)
        # 2. Expand to batch
        dr = rel_rows.unsqueeze(0).expand(B, -1, -1)  # [B,7,5]
        dc = rel_cols.unsqueeze(0).expand(B, -1, -1)  # [B,7,5]
        # 3. Rotate
        delta_x = torch.empty_like(dr)
        delta_y = torch.empty_like(dc)
        mask_e = direction == 0
        mask_s = direction == 1
        mask_w = direction == 2
        mask_n = direction == 3

        delta_x[mask_e], delta_y[mask_e] = dr[mask_e], dc[mask_e]
        delta_x[mask_s], delta_y[mask_s] = -dc[mask_s], dr[mask_s]
        delta_x[mask_w], delta_y[mask_w] = -dr[mask_w], -dc[mask_w]
        delta_x[mask_n], delta_y[mask_n] = dc[mask_n], -dr[mask_n]

        # 4. Translate
        base_x = location[:, 0].view(B,1,1).expand(-1,H,W)
        base_y = location[:, 1].view(B,1,1).expand(-1,H,W)
        map_x = base_x + delta_x
        map_y = base_y + delta_y

        # 5. Valid mask
        valid = (map_x >= 0) & (map_x < self.map_width) & (map_y >= 0) & (map_y < self.map_height)

        return map_x.long(), map_y.long(), valid
    
    def rotate_map_memory_batch(self, updated_map_memory: torch.Tensor, agent_direction: torch.Tensor) -> torch.Tensor:
        """
        Rotates a batch of map memory tensors and adjusts wall channel orders based on agent direction.
        The goal is to rotate the map so the agent is always effectively facing "East" (right)
        after the spatial rotation. Wall channels are also circularly shifted to maintain
        consistency relative to the agent's new orientation.
        Args:
            updated_map_memory (torch.Tensor): The input map memory tensor of shape (B, C, H, W)
            agent_direction (torch.Tensor): A tensor of shape (B,) containing the original direction of each agent in the batch.

        Returns:
            torch.Tensor: The batch of rotated map memory tensors, shape (B, C, H, W).
        """
        batch_size = updated_map_memory.size(0)
        device = updated_map_memory.device

        # Ensure agent_direction is a LongTensor for indexing and on the correct device
        if not isinstance(agent_direction, torch.LongTensor):
            agent_direction = agent_direction.long()
        
        if agent_direction.device != device:
            agent_direction = agent_direction.to(device)

        k_values_for_rotation = agent_direction # [B,]

        # Pre-calculate all 4 possible 90-degree rotated versions of the map
        map_rot0 = updated_map_memory  # k=0
        map_rot1 = torch.rot90(updated_map_memory, k=1, dims=(2, 3)) # 90 deg CCW
        map_rot2 = torch.rot90(updated_map_memory, k=2, dims=(2, 3)) # 180 deg CCW
        map_rot3 = torch.rot90(updated_map_memory, k=3, dims=(2, 3)) # 270 deg CCW

        # Stack these rotated maps along a new dimension (dim=0)
        all_spatial_rotations = torch.stack([map_rot0, map_rot1, map_rot2, map_rot3], dim=0) # [4, B, C, H, W]
        batch_indices = torch.arange(batch_size, device=device)
        rotated_map_memory_spatial = all_spatial_rotations[k_values_for_rotation, batch_indices] # [B, C, H, W]

        num_wall_channels = 4 # Channels 4, 5, 6, 7
        # agent_direction: 0 (E) -> shifts = 0 -> effective_idx = 0
        # agent_direction: 1 (S) -> shifts = -1 -> effective_idx = 3
        # agent_direction: 2 (W) -> shifts = -2 -> effective_idx = 2
        # agent_direction: 3 (N) -> shifts = -3 -> effective_idx = 1
        effective_roll_idx = torch.remainder(-agent_direction, num_wall_channels) # Shape (B,)

        walls_slice_original = rotated_map_memory_spatial[:, 4:8, :, :]

        # Pre-calculate all 4 possible rolled versions of the wall channels
        walls_roll0 = walls_slice_original # Corresponds to effective_idx = 0 (0 shifts)
        walls_roll1 = torch.roll(walls_slice_original, shifts=1, dims=1) # Corresponds to effective_idx = 1 (1 shift)
        walls_roll2 = torch.roll(walls_slice_original, shifts=2, dims=1) # Corresponds to effective_idx = 2 (2 shifts)
        walls_roll3 = torch.roll(walls_slice_original, shifts=3, dims=1) # Corresponds to effective_idx = 3 (3 shifts)
        
        all_wall_rolls = torch.stack([walls_roll0, walls_roll1, walls_roll2, walls_roll3], dim=0) # [4, B, num_wall_channels, H, W]
        # Select the correct rolled wall slice for each item in the batch
        rolled_walls_slice = all_wall_rolls[effective_roll_idx, batch_indices] # [B, num_wall_channels, H, W]

        final_rotated_map_memory = rotated_map_memory_spatial.clone()
        final_rotated_map_memory[:, 4:8, :, :] = rolled_walls_slice
        return final_rotated_map_memory

    def forward(self, viewcone: torch.Tensor, direction: torch.Tensor, location: torch.Tensor, scout: torch.Tensor, step: torch.Tensor, map_memory: torch.Tensor):
        """
        Forward pass to action logits, state value, and updated map memory.
        
        Args:
            viewcone (torch.Tensor): Viewcone tensor of shape (B, 7, 5).
            direction (torch.Tensor): Agent's direction tensor of shape (B,).
                0: East, 1: South, 2: West, 3: North
            location (torch.Tensor): Agent's location tensor of shape (B, 2).
            scout (torch.Tensor): Scout tensor of shape (B,).
                0: Guard, 1: Scout
            step (torch.Tensor): Step tensor of shape (B,).
            map_memory (torch.Tensor): Map memory tensor of shape (B, map_channels, map_height, map_width).
                Map channels:
                0: Agent Current Position (need to clear previous position)
                1: No vision / Unknown (Initial mapstate)
                2: Recon Point
                3: Mission Point (if both channel 2 and 3 are 0, then it's free space)
                4: right wall (walls match original challenge bits)
                5: bottom wall
                6: left wall
                7: top wall
                8: visited by agent
                9: Scout spotted n turns ago: value=(100-n)/100
                10: Guard spotted n turns ago: value=(100-n)/100

        Returns:
            action_logits (torch.Tensor): Action logits of shape (B, num_actions).
            state_value (torch.Tensor): State value of shape (B, 1).
            updated_map_memory (torch.Tensor): Updated map memory tensor of shape (B, map_channels, map_height, map_width).
        """
        # Ensure all inputs are on the correct device
        device = next(self.parameters()).device # Get the device the model is on
        viewcone = viewcone.to(device)
        direction = direction.to(device)
        location = location.to(device)
        scout = scout.to(device)
        step = step.to(device)
        map_memory = map_memory.to(device)
        if len(direction.shape) > 1:
            direction = direction.argmax(dim=1)
        if len(scout.shape) > 1:
            scout = scout.argmax(dim=1)
        if len(step.shape) > 1:
            step = step.argmax(dim=1)
        
        B, H, W = viewcone.shape
        vc = viewcone.to(torch.uint8).unsqueeze(1)  # [B,1,7,5]
        mask = (1 << torch.arange(8, device=device, dtype=torch.uint8))[:, None, None]  # [8,1,1]
        bits = (vc & mask).ne(0).float() # [B,8,7,5]

        walls = bits[:, 4:8]
        arange = torch.arange(4, device=device)  # [4]
        idx = (arange.unsqueeze(0) - direction.unsqueeze(1)) % 4  # [B,4]
        idx_expanded = idx.view(B, 4, 1, 1).expand(-1, -1, H, W)
        rolled_walls = torch.gather(walls, dim=1, index=idx_expanded)  # [B,4,7,5]
        bits[:, 4:8] = rolled_walls

        # Update map memory
        updated_map_memory = map_memory.clone() # [B,11,16,16]
        # Clear previous agent position and update visited channel
        updated_map_memory[:, self.AGENT_POS_CHANNEL, :, :] = 0.0
        batch_idx = torch.arange(B, device=device)
        x_idx = location[:, 0].long()
        y_idx = location[:, 1].long()
        updated_map_memory[batch_idx, self.AGENT_POS_CHANNEL, y_idx, x_idx] = 1.0 # Mark current position
        updated_map_memory[batch_idx, self.VISITED_CHANNEL, y_idx, x_idx] = 1.0 # Mark as visited
        # Update scout and guard channels
        updated_map_memory[:, self.SCOUT_CHANNEL, :, :] = torch.clamp(updated_map_memory[:, self.SCOUT_CHANNEL, :, :] / 2, min=0.0) # Changed from V6
        updated_map_memory[:, self.GUARD_CHANNEL, :, :] = torch.clamp(updated_map_memory[:, self.GUARD_CHANNEL, :, :] / 2, min=0.0) # Changed from V6

        map_x, map_y, valid = self.batched_viewcone_to_map_coords(direction, location) # [B,7,5]
        # Flatten the valid mask and map coordinates for easier indexing
        valid_flat = valid.view(-1) # [B*7*5]
        map_x_flat = map_x.view(-1) # [B*7*5]
        map_y_flat = map_y.view(-1) # [B*7*5]
        # Create flattened batch indices corresponding to the viewcone points
        batch_idx_flat = torch.arange(B, device=device).unsqueeze(1).expand(-1, 7 * 5).reshape(-1) # [B*7*5]

        # Extract the last 2 bits from the viewcone for empty, recon, mission status
        last2bits = (viewcone.to(torch.uint8) & 0b11).view(-1)  # [B*7*5]
        mask_empty = (last2bits == 1) # [B*7*5]
        mask_recon = (last2bits == 2) # [B*7*5]
        mask_mission = (last2bits == 3) # [B*7*5]

        # Combine valid map coordinates mask with the viewcone bit masks
        mask_empty_and_valid = valid_flat & mask_empty
        mask_recon_and_valid = valid_flat & mask_recon
        mask_mission_and_valid = valid_flat & mask_mission

        # Get the indices in the flattened viewcone coordinates that are both valid and match the bit mask
        update_indices_empty = torch.where(mask_empty_and_valid)
        update_indices_recon = torch.where(mask_recon_and_valid)
        update_indices_mission = torch.where(mask_mission_and_valid)

        # Use these indices to get the corresponding batch, y, and x coordinates in the map memory
        update_batch_idx_empty = batch_idx_flat[update_indices_empty]
        update_y_idx_empty = map_y_flat[update_indices_empty]
        update_x_idx_empty = map_x_flat[update_indices_empty]

        update_batch_idx_recon = batch_idx_flat[update_indices_recon]
        update_y_idx_recon = map_y_flat[update_indices_recon]
        update_x_idx_recon = map_x_flat[update_indices_recon]

        update_batch_idx_mission = batch_idx_flat[update_indices_mission]
        update_y_idx_mission = map_y_flat[update_indices_mission]
        update_x_idx_mission = map_x_flat[update_indices_mission]

        # Update Map Memory based on viewcone observations

        # Clear Unknown, Recon, and Mission channels where the viewcone shows empty space
        updated_map_memory[update_batch_idx_empty, self.UNKNOWN_CHANNEL, update_y_idx_empty, update_x_idx_empty] = 0.0
        updated_map_memory[update_batch_idx_empty, self.RECON_CHANNEL, update_y_idx_empty, update_x_idx_empty] = 0.0
        updated_map_memory[update_batch_idx_empty, self.MISSION_CHANNEL, update_y_idx_empty, update_x_idx_empty] = 0.0

        # Set Recon points and clear Unknown where the viewcone shows a Recon point
        updated_map_memory[update_batch_idx_recon, self.RECON_CHANNEL, update_y_idx_recon, update_x_idx_recon] = 1.0
        updated_map_memory[update_batch_idx_recon, self.UNKNOWN_CHANNEL, update_y_idx_recon, update_x_idx_recon] = 0.0

        # Set Mission points and clear Unknown where the viewcone shows a Mission point
        updated_map_memory[update_batch_idx_mission, self.MISSION_CHANNEL, update_y_idx_mission, update_x_idx_mission] = 1.0
        updated_map_memory[update_batch_idx_mission, self.UNKNOWN_CHANNEL, update_y_idx_mission, update_x_idx_mission] = 0.0

        # Update Wall channels (bits 4 to 7)
        for bit in range(4, 8):
            # Get the mask for the current wall bit in the viewcone
            wall_bit_mask = bits[:, bit].reshape(-1).bool() # [B*7*5]
            # Combine with the valid map coordinates mask
            wall_mask_and_valid = valid_flat & wall_bit_mask
            # Get the indices for updating
            update_indices_wall = torch.where(wall_mask_and_valid)
            # Get the corresponding batch, y, and x coordinates
            update_batch_idx_wall = batch_idx_flat[update_indices_wall]
            update_y_idx_wall = map_y_flat[update_indices_wall]
            update_x_idx_wall = map_x_flat[update_indices_wall]
            # Set the wall channel to 1.0 at these locations
            updated_map_memory[update_batch_idx_wall, bit, update_y_idx_wall, update_x_idx_wall] = 1.0

        # Update Scout (ch=9) and Guard (ch=10) channels
        # Bit 2 in viewcone corresponds to Scout, Bit 3 corresponds to Guard
        scout_bit_mask = bits[:, 2].reshape(-1).bool() # [B*7*5]
        guard_bit_mask = bits[:, 3].reshape(-1).bool() # [B*7*5]

        scout_mask_and_valid = valid_flat & scout_bit_mask
        guard_mask_and_valid = valid_flat & guard_bit_mask

        update_indices_scout = torch.where(scout_mask_and_valid)
        update_indices_guard = torch.where(guard_mask_and_valid)

        update_batch_idx_scout = batch_idx_flat[update_indices_scout]
        update_y_idx_scout = map_y_flat[update_indices_scout]
        update_x_idx_scout = map_x_flat[update_indices_scout]
        # Set the Scout channel to 1.0 at these locations
        updated_map_memory[update_batch_idx_scout, self.SCOUT_CHANNEL, update_y_idx_scout, update_x_idx_scout] = 1.0

        update_batch_idx_guard = batch_idx_flat[update_indices_guard]
        update_y_idx_guard = map_y_flat[update_indices_guard]
        update_x_idx_guard = map_x_flat[update_indices_guard]
        # Set the Guard channel to 1.0 at these locations
        updated_map_memory[update_batch_idx_guard, self.GUARD_CHANNEL, update_y_idx_guard, update_x_idx_guard] = 1.0

        rotated_map_memory = self.rotate_map_memory_batch(updated_map_memory.clone(), direction)
        rotated_map_memory.to(device)

        # Process rotated map memory CNN with attention
        map_cnn_out = F.relu(self.map_cnn1(rotated_map_memory))
        map_cnn_out, attention1 = self.attention1(map_cnn_out)
        
        map_cnn_out = F.relu(self.map_cnn2(map_cnn_out))
        map_cnn_out = self.maxpool1(map_cnn_out)
        map_cnn_out, attention2 = self.attention2(map_cnn_out)
        
        map_cnn_out = F.relu(self.map_cnn3(map_cnn_out))
        map_cnn_out = F.relu(self.map_cnn4(map_cnn_out))
        map_cnn_out = self.maxpool2(map_cnn_out)
        map_cnn_out, attention3 = self.attention3(map_cnn_out)

        # Flatten the output of the CNN layers
        map_features = map_cnn_out.view(map_cnn_out.size(0), -1)
        map_features = self.layernorm_cnn(map_features) # Change layernorm to not include scout and step for V5

        scout_embedding = self.scout_embedding1(scout.long())
        # print(scout_embedding.shape)
        combined_features = torch.cat((map_features, scout_embedding, (step/100.).unsqueeze(1)), dim=1)

        # --- Actor (Policy) Head ---
        actor_out = F.relu(self.layernorm_actor1(self.fc_actor1(combined_features)))
        actor_out = F.relu(self.layernorm_actor2(self.fc_actor2(actor_out)))
        action_logits = self.fc_actor3(actor_out)

        # --- Critic (Value) Head ---
        critic_out = F.relu(self.layernorm_critic1(self.fc_critic1(combined_features)))
        state_value = self.fc_critic2(critic_out)

        # Store attention maps for visualization if needed
        attention_maps = (attention1, attention2, attention3)

        return action_logits, state_value, updated_map_memory #, attention_maps
    
    def actor_parameters(self):
        """Returns an iterator over the actor parameters."""
        return itertools.chain(
            self.fc_actor1.parameters(), 
            self.layernorm_actor1.parameters(),
            self.fc_actor2.parameters(), 
            self.layernorm_actor2.parameters(),
            self.fc_actor3.parameters()
        )

    def critic_parameters(self):
        """Returns an iterator over the critic parameters."""
        return itertools.chain(
            self.fc_critic1.parameters(), 
            self.layernorm_critic1.parameters(),
            self.fc_critic2.parameters()
        )
    
    def shared_parameters(self):
        """Returns an iterator over the shared parameters."""
        return itertools.chain(
            self.map_cnn1.parameters(),
            self.attention1.parameters(),
            self.map_cnn2.parameters(),
            self.attention2.parameters(),
            self.map_cnn3.parameters(),
            self.map_cnn4.parameters(),
            self.attention3.parameters(),
            self.layernorm_cnn.parameters(),
            self.scout_embedding1.parameters()
        )

