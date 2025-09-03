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
        
        # draw 
        if self.viewer != None:
            for env in self.envs:
                self._add_rectangle_borderline(env)

            cam_pos = gymapi.Vec3(15.0, 0.0, 3.0)
            cam_target = gymapi.Vec3(10.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
        ###### Reward Definition ######

        ###### Reward Definition ######
        self.debug_viz = True

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
        self._ref_motion_times += self.dt
        
        # 처음부터 다시 시작
        motion_lengths = self._motion_lib.get_motion_length(self._ref_motion_ids)
        exceeded_mask = self._ref_motion_times >= motion_lengths
        if exceeded_mask.any():
            print("motion end, start again: ", torch.sum(exceeded_mask).item())
            self._ref_motion_times[exceeded_mask] = 0.0
        
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
