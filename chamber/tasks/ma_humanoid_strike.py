# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from asyncio import shield
from dis import dis
import torch
import math

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

import chamber.tasks.ase_humanoid_base.humanoid_amp_task as humanoid_amp_task
from chamber.utils import torch_utils

from enum import Enum
from chamber.utils.torch_jit_utils import quat_diff_rad

class HumanoidStrike(humanoid_amp_task.HumanoidAMPTask): # tracking
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # Motion tracking state initialization
        state_init = cfg["env"].get("stateInit", "Start")
        self._state_init = self.StateInit[state_init] # self._state_init = HumanoidStrike.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"].get("hybridInitProb", 0.5)
        
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        # Load motion data for tracking
        motion_file = cfg['env']['motion_file']
        self._load_motion(motion_file)
        
        # Motion tracking variables
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self._ref_motion_ids = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._ref_motion_times = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        # Strike-specific body IDs (still needed for observations)
        strike_body_names = cfg["env"]["strikeBodyNames"]
        self._strike_body_ids = self._build_body_ids_tensor(self.envs[0], self.humanoid_handles[0], strike_body_names)
        force_body_names = cfg["env"]["forceBodies"]
        self._force_body_ids = self._build_body_ids_tensor(self.envs[0], self.humanoid_handles[0], force_body_names)
        
        
        if self.viewer != None:
            for env in self.envs:
                self._add_rectangle_borderline(env)

            cam_pos = gymapi.Vec3(15.0, 0.0, 3.0)
            cam_target = gymapi.Vec3(10.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
        ###### Reward Definition ######

        ###### Reward Definition ######

        return
    
    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            # Motion tracking observations: target poses, velocities. # self.num_dof = 31
            # target root pos + rot + vel + ang_vel + target dof pos + vel
            obs_size = 3 + 6 + 3 + 3 + 31 * 2
        return obs_size

    def _create_envs(self, num_envs, spacing, num_per_row):
        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset, humanoid_asset_op):
        super()._build_env(env_id, env_ptr, humanoid_asset, humanoid_asset_op)
        return

    def _build_body_ids_tensor(self, env_ptr, actor_handle, body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    # reset act
    def _reset_actors(self, env_ids):
        if (self._state_init == self.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == self.StateInit.Start
              or self._state_init == self.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == self.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return

    def _reset_default(self, env_ids):
        # Default reset for non-motion tracking initialization
        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        self._dof_pos[env_ids] = tensor_clamp(self._initial_dof_pos[env_ids] + positions, self.dof_limits_lower, self.dof_limits_upper)
        self._dof_vel[env_ids] = velocities

        self._dof_pos_op[env_ids] = tensor_clamp(self._initial_dof_pos[env_ids] + positions, self.dof_limits_lower, self.dof_limits_upper)
        self._dof_vel_op[env_ids] = velocities

        agent_env_ids = expand_env_ids(env_ids, 2)
        self._humanoid_root_states[agent_env_ids] = self._initial_humanoid_root_states[agent_env_ids]
        
        self._reset_default_env_ids = env_ids
        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        
        if (self._state_init == HumanoidStrike.StateInit.Random or self._state_init == HumanoidStrike.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self._state_init == HumanoidStrike.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        # Store reference motion info for tracking
        self._ref_motion_ids[env_ids] = motion_ids
        self._ref_motion_times[env_ids] = motion_times
        
        self._reset_ref_env_ids = env_ids
        return
    
    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)

        return
    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        # Set only the first humanoid (ego agent) to reference motion
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        
        # Set opponent to default position for now
        agent_env_ids_op = env_ids + self.num_envs
        self._humanoid_root_states[agent_env_ids_op] = self._initial_humanoid_root_states[agent_env_ids_op]
        self._dof_pos_op[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel_op[env_ids] = self._initial_dof_vel[env_ids]
        
        return

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        return

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._ref_motion_times += self.dt # import pdb; pdb.set_trace()
        return

    def post_physics_step(self):
        super().post_physics_step()

    def _compute_observations(self):
        obs, obs_op = self._compute_humanoid_obs()
        if (self._enable_task_obs):
            task_obs, task_obs_op = self._compute_task_obs()
            obs = torch.cat([obs, task_obs], dim=-1)
            obs_op = torch.cat([obs_op, task_obs_op], dim=-1)
        self.obs_buf[:self.num_envs] = obs
        self.obs_buf[self.num_envs:] = obs_op
        return

    def _compute_task_obs(self):
        # Get target pose from reference motion
        root_states = self._humanoid_root_states[self.humanoid_indices]
        
        # Get reference motion state for current time + small lookahead
        future_times = self._ref_motion_times + 0.1  # 100ms lookahead
        ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_key_pos = \
            self._motion_lib.get_motion_state(self._ref_motion_ids, future_times)

        # Compute relative target information
        current_root_pos = root_states[:, 0:3]
        current_root_rot = root_states[:, 3:7]
        
        # Target relative position and rotation
        heading_rot = torch_utils.calc_heading_quat_inv(current_root_rot)
        target_rel_pos = quat_rotate(heading_rot, ref_root_pos - current_root_pos)
        target_rel_rot = quat_mul(heading_rot, ref_root_rot)
        target_rel_rot_obs = torch_utils.quat_to_tan_norm(target_rel_rot)
        
        # Target velocities in local frame
        target_rel_vel = quat_rotate(heading_rot, ref_root_vel)
        target_rel_ang_vel = quat_rotate(heading_rot, ref_root_ang_vel)
        
        # Target DOF positions and velocities (relative)
        current_dof_pos = self._dof_pos
        target_rel_dof_pos = ref_dof_pos - current_dof_pos
        target_rel_dof_vel = ref_dof_vel - self._dof_vel

        obs = torch.cat([target_rel_pos, target_rel_rot_obs, target_rel_vel, target_rel_ang_vel,
                        target_rel_dof_pos, target_rel_dof_vel], dim=-1)
        
        # For opponent, provide zero observations for now (or could track different motion) TODO
        obs_op = torch.zeros_like(obs)
        
        return obs, obs_op

    def _compute_reward(self, actions):
        root_states = self._humanoid_root_states[self.humanoid_indices]
        
        # Get current reference motion state
        ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_key_pos = \
            self._motion_lib.get_motion_state(self._ref_motion_ids, self._ref_motion_times)
        
        body_pos = self._rigid_body_pos
        body_vel = self._rigid_body_vel
        
        self.rew_buf[:] = compute_motion_tracking_reward(
            root_states=root_states,
            body_pos=body_pos,
            body_vel=body_vel,
            dof_pos=self._dof_pos,
            dof_vel=self._dof_vel,
            ref_root_pos=ref_root_pos,
            ref_root_rot=ref_root_rot,
            ref_dof_pos=ref_dof_pos,
            ref_root_vel=ref_root_vel,
            ref_root_ang_vel=ref_root_ang_vel,
            ref_dof_vel=ref_dof_vel,
            ref_key_pos=ref_key_pos,
            key_body_ids=self._key_body_ids
        )
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf, 
            self.progress_buf,
            self._contact_forces, 
            self._contact_body_ids,
            self._rigid_body_pos,
            self.max_episode_length,
            self._enable_early_termination,
            self._termination_heights
        )
        return

#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
# def compute_strike_observations(root_states, root_states_op, body_pos, body_rot,
#                                 body_pos_op, body_vel_op, borderline,
#                                 ):
#     # type: (Tensor, Tensor, Tensor, Tensor, Tensor,Tensor,float) -> Tensor
#     root_pos = root_states[:, 0:3]
#     root_rot = root_states[:, 3:7]
#     ego_sword_pos = body_pos[:, 6, :]
#     ego_sword_rot = body_rot[:, 6, :]
#     ego_shield_pos = body_pos[:, 9, :]
#     ego_shield_rot = body_rot[:, 9, :]

#     root_pos_op = root_states_op[:, 0:3]
#     root_rot_op = root_states_op[:, 3:7]
#     root_vel_op = root_states_op[:, 7:10]
#     root_ang_op = root_states_op[:, 10:13]
#     op_sword_pos = body_pos_op[:, 6, :]
#     op_sword_vel = body_vel_op[:, 6, :]
#     op_torso_pos = body_pos_op[:, 1, :]
#     op_torso_vel = body_vel_op[:, 1, :]
#     op_head_pos = body_pos_op[:, 2, :]
#     op_head_vel = body_vel_op[:, 2, :]
#     op_right_upper_arm_pos = body_pos_op[:, 3, :]
#     op_right_thigh_pos = body_pos_op[:, 11, :]
#     op_left_thigh_pos = body_pos_op[:, 14, :]

#     ##*******************************************************##
#     relative_x_1 =  borderline - root_pos[:, 0]
#     relative_x_2 = root_pos[:, 0] + borderline
#     relative_x = torch.minimum(relative_x_1, relative_x_2)
#     relative_x = torch.unsqueeze(relative_x, -1)
#     relative_y_1 =  borderline - root_pos[:, 1]
#     relative_y_2 = root_pos[:,1] + borderline
#     relative_y = torch.minimum(relative_y_1, relative_y_2)
#     relative_y = torch.unsqueeze(relative_y, -1)
#     ##*******************************************************##

#     heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
#     sword_rot = torch_utils.calc_heading_quat_inv(ego_sword_rot)
#     shield_rot = torch_utils.calc_heading_quat_inv(ego_shield_rot)

#     local_op_relative_pos = root_pos_op - root_pos
#     local_op_relative_pos[..., -1] = root_pos_op[..., -1]
#     local_op_relative_pos = quat_rotate(heading_rot, local_op_relative_pos)

#     local_op_vel = quat_rotate(heading_rot, root_vel_op)
#     local_op_ang_vel = quat_rotate(heading_rot, root_ang_op)

#     local_op_rot = quat_mul(heading_rot, root_rot_op)
#     local_op_rot_obs = torch_utils.quat_to_tan_norm(local_op_rot)
#     ##*******************************************************##

#     # op sword relative ego position and vel
#     local_op_relative_sword_pos = op_sword_pos - root_pos
#     local_op_relative_sword_pos = quat_rotate(heading_rot, local_op_relative_sword_pos)
#     local_op_sword_vel = quat_rotate(heading_rot, op_sword_vel)
    
#     # op sword relative ego shield position and vel
#     local_op_sword_shield_pos = op_sword_pos - ego_shield_pos
#     local_op_sword_shield_pos = quat_rotate(shield_rot, local_op_sword_shield_pos)
#     local_op_sword_shield_vel = quat_rotate(shield_rot, op_sword_vel)
    
#     # relative position and vel of ego sword and op up body
#     relative_sword_torso_pos = op_torso_pos - ego_sword_pos
#     relative_sword_torso_pos = quat_rotate(sword_rot, relative_sword_torso_pos)
#     relative_sword_torso_vel = quat_rotate(sword_rot, op_torso_vel)
#     relative_sword_head_pos = op_head_pos - ego_sword_pos
#     relative_sword_head_pos = quat_rotate(sword_rot, relative_sword_head_pos)
#     relative_sword_head_vel = quat_rotate(sword_rot, op_head_vel)
#     relative_sword_right_arm_pos = op_right_upper_arm_pos - ego_sword_pos
#     relative_sword_right_arm_pos = quat_rotate(sword_rot, relative_sword_right_arm_pos)
#     relative_sword_right_thigh_pos = op_right_thigh_pos - ego_sword_pos
#     relative_sword_right_thigh_pos = quat_rotate(sword_rot, relative_sword_right_thigh_pos)
#     relative_sword_left_thigh_pos = op_left_thigh_pos - ego_sword_pos
#     relative_sword_left_thigh_pos = quat_rotate(sword_rot, relative_sword_left_thigh_pos)

#     obs = torch.cat([relative_x, relative_y,
#                      local_op_relative_pos, local_op_rot_obs,
#                      local_op_vel, local_op_ang_vel,
#                      local_op_relative_sword_pos, local_op_sword_vel,
#                      local_op_sword_shield_pos, local_op_sword_shield_vel,
#                      relative_sword_torso_pos, relative_sword_torso_vel,
#                      relative_sword_head_pos, relative_sword_head_vel,
#                      relative_sword_right_arm_pos, relative_sword_right_thigh_pos,
#                      relative_sword_left_thigh_pos
#                      ], dim=-1)
#     return obs

# @torch.jit.script
# def compute_strike_reward(root_states, root_states_op, body_pos, body_ang_vel,
#                           prev_body_ang_vel, body_vel, prev_body_vel,
#                           body_pos_op, force_body_ids, strike_body_ids,
#                           contact_force, contact_force_op, contact_body_ids,
#                           borderline, termination_heights, dt):
#     # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor,Tensor,Tensor,float, Tensor, float) -> Tuple[Tensor, Tensor,Tensor]

#     op_fall_reward_w = 200.0
#     ego_fall_out_reward_w = 50.0
#     shield_to_sword_pos_reward_w = 1.0
#     damage_reward_w = 8.0
#     sword_to_op_reward_w = 0.8
#     reward_energy_w = 3.0
#     reward_strike_vel_acc_w = 3.0
#     reward_face_w = 4.0
#     reward_foot_to_op_w = 10.0
#     reward_kick_w = 2.0

#     num_envs = root_states.shape[0]
#     reward = torch.zeros((num_envs, 1), dtype=torch.float32)
#     root_xy_pos = root_states[:, 0:2]
#     root_pos = root_states[:, 0:3]
#     ego_sword_pos = body_pos[:, 6, 0:3]
#     ego_shield_pos = body_pos[:, 9, 0:3]
#     ego_right_foot_pos = body_pos[:, 13, 0:3]
#     op_sword_pos = body_pos_op[:, 6, 0:3]
#     op_torse_pos = body_pos_op[:, 1, 0:3]
#     op_right_thigh_pos = body_pos_op[:, 11, 0:3]
#     op_left_thigh_pos = body_pos_op[:, 14, 0:3]
#     root_pos_xy_op = root_states_op[:, 0:2]
#     root_pos_xy = root_states[:, 0:2]
#     root_pos_op = root_states_op[:, 0:3]
#     root_rot = root_states[:, 3:7]
#     root_rot_op = root_states_op[:, 3:7]
#     up = torch.zeros_like(root_pos_op)
#     up[..., -1] = 1
#     contact_buf = contact_force.clone()
#     contact_buf_op = contact_force_op.clone()

#     ##*****************r energy******************##
#     strike_body_vel = body_vel[:, strike_body_ids, :]
#     strike_body_vel_norm = torch.sum(torch.norm(strike_body_vel, dim=-1), dim=1)
#     strike_body_vel_norm = torch.clamp(strike_body_vel_norm, max=20)
#     distance = root_pos_xy_op - root_xy_pos
#     distance = torch.norm(distance, dim=-1)
#     zeros = torch.zeros_like(distance)
#     k_dist = torch.exp(-10 * torch.maximum(zeros, distance - 2.0))
#     r_energy = k_dist * strike_body_vel_norm
#     r_energy = r_energy / 20.
    
#     strike_vel_dfff = body_vel[:, strike_body_ids, :] - prev_body_vel[:, strike_body_ids, :]
#     strike_vel_acc = strike_vel_dfff / dt
#     strike_vel_acc = torch.sum(torch.norm(strike_vel_acc, dim=-1), dim=1)
#     strike_vel_acc = torch.clamp(strike_vel_acc, max=1000)
#     strike_vel_acc = k_dist * strike_vel_acc / 500
#     r_strike_vel_acc = strike_vel_acc
#     ##*****************r damage******************##
#     ego_to_op_force = contact_buf_op[:, force_body_ids, :]

#     op_to_ego_force = contact_buf[:, force_body_ids, :]

#     force_ego_to_op = torch.norm(ego_to_op_force, dim=2).sum(dim=1)

#     force_op_to_ego = torch.norm(op_to_ego_force, dim=2).sum(dim=1)

#     r_damage = force_ego_to_op - force_op_to_ego * 2
#     r_damage = torch.clamp(r_damage, min= -200.)
#     r_damage /= 100

#     ##*****************r kick******************##
#     ego_foot_op_torse_distance = op_torse_pos - ego_right_foot_pos
#     ego_foot_op_torse_err = torch.norm(ego_foot_op_torse_distance, dim=-1)
#     succ_foot = ego_foot_op_torse_err < 0.1
#     r_foot_to_op = torch.exp(-0.5 * ego_foot_op_torse_err)
#     constant_r = torch.ones_like(r_foot_to_op)
#     r_foot_to_op = torch.where(succ_foot, constant_r, r_foot_to_op)
    
#     foot_height = ego_right_foot_pos[..., 2]
#     succ_kick = foot_height >= 0.4
#     zeros = torch.zeros_like(succ_kick)
#     constant_r_kick = torch.ones_like(succ_kick)
#     r_kick = torch.where(succ_kick, constant_r_kick, foot_height)
    
#     ##*****************r close******************##
#     # sword -> torso
#     pos_err_scale1 = 1.0
#     pos_err_scale2 = 2.0

#     sword_torse_distance = op_torse_pos - ego_sword_pos
#     sword_torse_err = torch.sum(sword_torse_distance * sword_torse_distance, dim=-1)

#     sword_right_thigh_distance = op_right_thigh_pos - ego_sword_pos
#     sword_right_thigh_err = torch.sum(sword_right_thigh_distance * sword_right_thigh_distance, dim=-1)

#     sword_left_thigh_distance = op_left_thigh_pos - ego_sword_pos
#     sword_left_thigh_err = torch.sum(sword_left_thigh_distance * sword_left_thigh_distance, dim=-1)

#     sword_sword_distance = op_sword_pos - ego_sword_pos
#     sword_sword_err = torch.sum(sword_sword_distance * sword_sword_distance, dim=-1)
    
#     # zeros = torch.zeros_like(sword_torse_distance)
#     r_close = torch.exp(-pos_err_scale1 * sword_torse_err) # -> [0, 1]
#     r_close += torch.exp(-pos_err_scale1 * sword_right_thigh_err)
#     r_close += torch.exp(-pos_err_scale1 * sword_left_thigh_err)
#     r_close += torch.exp(-pos_err_scale2 * sword_sword_err)
#     ##*****************r shelid with op sword******************##
#     pos_err_scale3 = 2.0
#     ego_shield_op_sword_distance = op_sword_pos - ego_shield_pos
#     ego_shield_op_sword_err = torch.sum(ego_shield_op_sword_distance * ego_shield_op_sword_distance, dim=-1)
#     r_shield_to_sword = torch.exp(-pos_err_scale3 * ego_shield_op_sword_err)

#     ##*****************r face******************##
#     tar_dir = root_pos_xy_op - root_xy_pos
#     tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)

#     heading_rot = torch_utils.calc_heading_quat(root_rot)
#     facing_dir = torch.zeros_like(root_pos)
#     facing_dir[..., 0] = 1.0
#     facing_dir = quat_rotate(heading_rot, facing_dir)
#     facing_err = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
#     facing_reward = torch.clamp_min(facing_err, 0.0)

#     ##*****************r op fall******************##
#     masked_contact_buf_op = contact_buf_op.clone()
#     masked_contact_buf_op[:, contact_body_ids, :] = 0
#     fall_contact_op = torch.any(torch.abs(masked_contact_buf_op) > 0.1, dim=-1)
#     fall_contact_op = torch.any(fall_contact_op, dim=-1)

#     body_height_op = body_pos_op[..., 2]
#     fall_height_op = body_height_op < termination_heights
#     fall_height_op[:, contact_body_ids] = False
#     fall_height_op = torch.any(fall_height_op, dim=-1)
#     has_fallen_op = torch.logical_and(fall_contact_op, fall_height_op)

#     op_up = quat_rotate(root_rot_op, up)
#     op_rot_err = torch.sum(up * op_up, dim=-1)
#     op_rot_r = 0.6 * torch.clamp_min(1.0 - op_rot_err, 0.0) # -> [0, 1] succ = op_rot_err < 0.2
#     op_rot_r = torch.where(has_fallen_op, torch.ones_like(op_rot_r), op_rot_r)

#     # test, when op fall, then r_close = 0 to encourage to agents separate.
#     r_separate = torch.norm((root_pos_xy_op - root_pos_xy), dim=-1)
#     r_separate = torch.where(r_separate > 0.1, r_separate, torch.zeros_like(r_separate))
#     r_close = torch.where(has_fallen_op, r_separate, r_close)
#     r_shield_to_sword = torch.where(has_fallen_op, torch.zeros_like(r_shield_to_sword), r_shield_to_sword)
    
#     ##*****************r penalty******************##
#     relative_x_1 =  borderline - root_xy_pos[:, 0]
#     relative_x_2 = root_xy_pos[:, 0] + borderline
#     relative_x = torch.minimum(relative_x_1, relative_x_2)
#     relative_x = relative_x < 0
#     relative_y_1 =  borderline - root_xy_pos[:, 1]
#     relative_y_2 = root_xy_pos[:,1] + borderline
#     relative_y = torch.minimum(relative_y_1, relative_y_2)
#     relative_y = relative_y < 0
#     is_out = relative_x | relative_y
#     r_penalty = is_out * 1.0

#     masked_contact_buf = contact_force.clone()
#     masked_contact_buf[:, contact_body_ids, :] = 0
#     fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
#     fall_contact = torch.any(fall_contact, dim=-1)
#     body_height = body_pos[..., 2]
#     fall_height = body_height < termination_heights  
#     fall_height[:, contact_body_ids] = False
#     fall_height = torch.any(fall_height, dim=-1)
#     has_fallen_ego = torch.logical_and(fall_contact, fall_height)
#     r_penalty += has_fallen_ego * 1.0

#     ##*****************r penalty******************##
#     reward = -r_penalty * ego_fall_out_reward_w + op_rot_r * op_fall_reward_w + \
#         r_shield_to_sword * shield_to_sword_pos_reward_w + r_close * sword_to_op_reward_w +\
#             r_damage * damage_reward_w + r_energy * reward_energy_w + facing_reward * reward_face_w + \
#                 r_strike_vel_acc * reward_strike_vel_acc_w + r_foot_to_op * reward_foot_to_op_w +\
#                     r_kick * reward_kick_w

#     return reward, force_ego_to_op, force_op_to_ego


# @torch.jit.script
# def compute_humanoid_reset(reset_buf, progress_buf, ego_to_op_damage, op_to_ego_damage,
#                            contact_buf, contact_buf_op, contact_body_ids,
#                            rigid_body_pos, rigid_body_pos_op, max_episode_length,
#                            enable_early_termination, termination_heights, borderline):
#     # type: (Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, float) -> Tuple[Tensor, Tensor,Tensor,Tensor,Tensor]

#     terminated = torch.zeros_like(reset_buf)

#     if (enable_early_termination):
#         masked_contact_buf = contact_buf.clone()
#         masked_contact_buf_op = contact_buf_op.clone()
#         masked_contact_buf[:, contact_body_ids, :] = 0
#         masked_contact_buf_op[:, contact_body_ids, :] = 0
#         fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
#         fall_contact = torch.any(fall_contact, dim=-1)
#         fall_contact_op = torch.any(torch.abs(masked_contact_buf_op) > 0.1, dim=-1)
#         fall_contact_op = torch.any(fall_contact_op, dim=-1)

#         body_height = rigid_body_pos[..., 2]
#         body_height_op = rigid_body_pos_op[..., 2]
#         fall_height = body_height < termination_heights
#         fall_height_op = body_height_op < termination_heights
#         fall_height[:, contact_body_ids] = False
#         fall_height_op[:, contact_body_ids] = False
#         fall_height = torch.any(fall_height, dim=-1)
#         fall_height_op = torch.any(fall_height_op, dim=-1)

#         ## out area
#         root_pos = rigid_body_pos[:, 0, 0:2]
#         root_pos_op = rigid_body_pos_op[:, 0, 0:2]
#         relative_x_1 =  borderline - root_pos[:, 0]
#         relative_x_2 = root_pos[:, 0] + borderline
#         relative_x = torch.minimum(relative_x_1, relative_x_2)
#         relative_x = relative_x < 0
#         relative_y_1 =  borderline - root_pos[:, 1]
#         relative_y_2 = root_pos[:,1] + borderline
#         relative_y = torch.minimum(relative_y_1, relative_y_2)
#         relative_y = relative_y < 0
#         is_out_ego = relative_x | relative_y

#         relative_x_1_op =  borderline - root_pos_op[:, 0]
#         relative_x_2_op = root_pos_op[:, 0] + borderline
#         relative_x_op = torch.minimum(relative_x_1_op, relative_x_2_op)
#         relative_x_op = relative_x_op < 0
#         relative_y_1_op =  borderline - root_pos_op[:, 1]
#         relative_y_2_op = root_pos_op[:,1] + borderline
#         relative_y_op = torch.minimum(relative_y_1_op, relative_y_2_op)
#         relative_y_op = relative_y_op < 0
#         is_out_op = relative_x_op | relative_y_op

#         is_out = is_out_ego | is_out_op
        
#         has_failed = is_out

#         # first timestep can sometimes still have nonzero contact forces
#         # so only check after first couple of steps
#         has_failed *= (progress_buf > 1)

#         terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)
#     damage_ego_more_than_op = ego_to_op_damage > op_to_ego_damage
#     damage_op_more_than_ego = op_to_ego_damage > ego_to_op_damage

#     reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)
#     win = torch.where(reset, damage_ego_more_than_op, torch.zeros_like(reset_buf, dtype=torch.bool))
#     lose = torch.where(reset, damage_op_more_than_ego, torch.zeros_like(reset_buf, dtype=torch.bool))
#     draw = torch.where(reset, ego_to_op_damage == op_to_ego_damage, torch.zeros_like(reset_buf, dtype=torch.bool))

#     return reset, terminated, win, lose, draw

@torch.jit.script
def compute_motion_tracking_reward(root_states, body_pos, body_vel, dof_pos, dof_vel,
                                  ref_root_pos, ref_root_rot, ref_dof_pos, 
                                  ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_key_pos,
                                  key_body_ids):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    
    # Reward weights
    w_root_pos = 2.0
    w_root_rot = 1.0
    w_root_vel = 0.5
    w_root_ang_vel = 0.5
    w_dof_pos = 1.0
    w_dof_vel = 0.1
    w_key_pos = 2.0
    
    num_envs = root_states.shape[0]
    
    # Root position reward
    root_pos_err = torch.sum(torch.square(root_states[:, 0:3] - ref_root_pos), dim=-1)
    root_pos_reward = torch.exp(-2.0 * root_pos_err)
    
    # Root rotation reward  
    root_rot = root_states[:, 3:7]
    root_rot_err = quat_diff_rad(root_rot, ref_root_rot)
    root_rot_reward = torch.exp(-1.0 * root_rot_err)
    
    # Root velocity reward
    root_vel = root_states[:, 7:10]
    root_vel_err = torch.sum(torch.square(root_vel - ref_root_vel), dim=-1)
    root_vel_reward = torch.exp(-0.5 * root_vel_err)
    
    # Root angular velocity reward
    root_ang_vel = root_states[:, 10:13]
    root_ang_vel_err = torch.sum(torch.square(root_ang_vel - ref_root_ang_vel), dim=-1)
    root_ang_vel_reward = torch.exp(-0.1 * root_ang_vel_err)
    
    # DOF position reward
    dof_pos_err = torch.sum(torch.square(dof_pos - ref_dof_pos), dim=-1)
    dof_pos_reward = torch.exp(-2.0 * dof_pos_err)
    
    # DOF velocity reward
    dof_vel_err = torch.sum(torch.square(dof_vel - ref_dof_vel), dim=-1)
    dof_vel_reward = torch.exp(-0.05 * dof_vel_err)
    
    # Key body position reward
    key_body_pos = body_pos[:, key_body_ids, :]
    key_pos_err = torch.sum(torch.square(key_body_pos - ref_key_pos), dim=(-1, -2))
    key_pos_reward = torch.exp(-5.0 * key_pos_err)
    
    # Combined reward
    reward = (w_root_pos * root_pos_reward + 
              w_root_rot * root_rot_reward +
              w_root_vel * root_vel_reward +
              w_root_ang_vel * root_ang_vel_reward +
              w_dof_pos * dof_pos_reward +
              w_dof_vel * dof_vel_reward +
              w_key_pos * key_pos_reward)
              
    return reward

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids,
                          rigid_body_pos, max_episode_length, enable_early_termination, 
                          termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]

    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        # Check for falls
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_failed = torch.logical_and(fall_contact, fall_height)
        has_failed *= (progress_buf > 1)

        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)
    
    return reset, terminated

@torch.jit.script
def expand_env_ids(env_ids, n_agents):
    # type: (Tensor, int) -> Tensor
    device = env_ids.device
    agent_env_ids = torch.zeros((n_agents * len(env_ids)), device=device, dtype=torch.long)
    for idx in range(n_agents):
        agent_env_ids[idx::n_agents] = env_ids * n_agents + idx
    return agent_env_ids
