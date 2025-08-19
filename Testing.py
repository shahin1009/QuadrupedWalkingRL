import os
# os.environ['MUJOCO_GL'] = 'egl'
import mujoco
from mujoco import Renderer
import numpy as np
import time
from IPython.display import HTML, display
from IPython.display import Image as IPImage, display
from PIL import Image
import matplotlib.pyplot as plt

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback , EvalCallback
from stable_baselines3.common.monitor import Monitor
import os

from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from pathlib import Path

import os



DEFAULT_CAMERA_CONFIG = {
    "azimuth": 120.0,
    "distance": 3.0,
    "elevation": -25.0,
    "lookat": np.array([0., 0., 0.]),
    "fixedcamid": 0,
    "trackbodyid": -1,
    "type": 2,
}


class QuadrupedEnv(MujocoEnv):
    metadata = {'render.modes': ['human'],'render_fps': 25}
    def __init__(self, ctrl_type="position",**kwargs):

        model_path = Path("unitree_a1/scene.xml")
        MujocoEnv.__init__(
            self,
            model_path=model_path.absolute().as_posix(),
            frame_skip=20,  # Perform an action every 10 frames (dt(=0.002) * 50
            observation_space=None,  # Manually set afterwards
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.renderer = mujoco.Renderer(self.model, 480, 480)
        self.cam = mujoco.MjvCamera()


        # self._default_joint_position = np.array(self.model.key_ctrl[0])
        # print(self._default_joint_position)
        self._default_joint_position = np.array([-0.03, 0.9, -1.4,
                                                 0.03, 0.9, -1.4,
                                                 -0.03, 0.9, -1.4,
                                                 0.03, 0.9, -1.4], dtype=np.float32)
        # self._default_joint_position = np.array(self.model.key_ctrl[0])
        # Observation: base vel (3), ori (4), joint angles (12), joint velocities (12),previous_actions (12) = 43
        obs_dim = 46
        obs_high = np.array([np.inf]*obs_dim*2, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self._previous_observation = np.zeros(obs_dim, dtype=np.float32)


        self.joint_limits_low = np.array([
            -0.03, 0.25, -1.6,  # Front left leg
            -0.03, 0.25, -1.6,  # Front right leg
            -0.03, 0.25, -1.6,  # Rear left leg
            -0.03, 0.25, -1.6   # Rear right leg
        ], dtype=np.float32)

        self.joint_limits_high = np.array([
            0.03, 1.2, -1.1,   # Front left leg
            0.03, 1.2, -1.1,   # Front right leg
            0.03, 1.2, -1.1,   # Rear left leg
            0.03, 1.2, -1.1    # Rear right leg
        ], dtype=np.float32)


        self.action_space = spaces.Box(
            low=self.joint_limits_low,
            high=self.joint_limits_high,
            shape=(12,),
            dtype=np.float32
        )

        self._gravity_vector = np.array(self.model.opt.gravity)

        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)

        self.episode_reward_ = 0.0
        self.tracking_lin_vel_reward_ = 0
        self.tracking_ang_vel_reward_ = 0
        self.helthy_reward_ = 0
        self.feet_air_time_reward_ = 0

        self.torque_cost_ = 0
        self.action_diff_penalty_ = 0
        self.lin_vel_z_penalty_ = 0
        self.xy_angular_velocity_cost_ = 0
        self.action_sym_ = 0
        self.acceleration_cost_ = 0
        self.orientation_penalty_ = 0
        self.default_joint_position_cost_ = 0



        self.step_counter = 0
        self.episode_counter = 0
        self.log_episode_count =  20
        self.prev_x = 0.0
        self.reached_target = False


        self.goal_distance = 9.0
        self.tracking_sigma = 0.25
        self.maximum_episode_steps = 1024
        self._max_episode_time_sec = 15.0
        self._curriculum_base = 0.3

        self.prev_action = np.zeros_like(self._default_joint_position)

        self.target_lin_vel = self.set_target_velocity()
        self.target_ang_vel = 0.0  # yaw rate in rad/s


        self._healthy_z_range = (0.3, 0.345)
        self._healthy_pitch_range = (-np.deg2rad(15), np.deg2rad(15))
        self._healthy_roll_range = (-np.deg2rad(15), np.deg2rad(15))

        self._cfrc_ext_feet_indices = [4, 7, 10, 13]


        feet_site = [
            "FR",
            "FL",
            "RR",
            "RL",
        ]
        self._feet_site_name_to_id = {
            f: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        }

        self._main_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY.value, "trunk"
        )

        self._gravity_vector = np.array(self.model.opt.gravity)



        dof_position_limit_multiplier = 0.9  # The % of the range that is not penalized
        ctrl_range_offset = (
            0.5
            * (1 - dof_position_limit_multiplier)
            * (
                self.model.actuator_ctrlrange[:, 1]
                - self.model.actuator_ctrlrange[:, 0]
            )
        )
        # First value is the root joint, so we ignore it
        self._soft_joint_range = np.copy(self.model.actuator_ctrlrange)
        self._soft_joint_range[:, 0] += ctrl_range_offset
        self._soft_joint_range[:, 1] -= ctrl_range_offset

        self.reward_weights={}
        self.cost_weights={}

    def set_weights(self, reward_weights, cost_weights):
        self.reward_weights = reward_weights
        self.cost_weights = cost_weights




    def set_target_velocity(self):
        # V_x_rand = np.random.uniform(0.4, 0.7)
        return np.array([0.3, 0.0])


    def reset(self,seed=None,options=None):
      super().reset(seed=seed)
      self.reset_model()

      self.prev_x = 0.0
      self.episode_reward_ = 0.0
      self.tracking_lin_vel_reward_ = 0
      self.tracking_ang_vel_reward_ = 0
      self.helthy_reward_ = 0
      self.feet_air_time_reward_ = 0

      self.torque_cost_ = 0
      self.action_diff_penalty_ = 0
      self.lin_vel_z_penalty_ = 0
      self.xy_angular_velocity_cost_ = 0
      self.action_sym_ = 0
      self.acceleration_cost_ = 0
      self.orientation_penalty_ = 0
      self.default_joint_position_cost_ = 0
      self.target_lin_vel = self.set_target_velocity()

      current_obs = self._get_obs()
      self._previous_observation = np.zeros_like(current_obs)
      full_obs = np.concatenate([current_obs, self._previous_observation])
      self._previous_observation = current_obs.copy()

      return full_obs,{}

    def reset_model(self):
        self.step_counter = 0
        self.data.qpos[0:3] = [0.0, 0.0, 0.3]
        # self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        # self.data.qpos[3:7] = [0.0, 1.0, 0.0, 0.0]

        self.data.qpos[7:19] = self._default_joint_position.copy()

        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)
        self.prev_action = self._default_joint_position.copy()


    def step(self, action):
      self.step_counter += 1
      ALPHA = 0.7
      action_filtered = ALPHA * self.prev_action + (1 - ALPHA) * action
      if self.step_counter <1:
            action_filtered = self._default_joint_position.copy() 
      self.do_simulation(action_filtered, self.frame_skip)

      current_obs = self._get_obs()
      obs = np.concatenate([current_obs, self._previous_observation])




      reward = self._compute_reward(current_obs, action)

      terminated = not self.is_healthy

      truncated = self.step_counter >= self.maximum_episode_steps

      info={'terminated': terminated,
                'truncated:': truncated,}
      self.prev_action = action_filtered
      self._previous_observation = current_obs.copy()


      if terminated or truncated:
            self.episode_counter += 1
            info = {
                "total_reward": self.episode_reward_,
                "tracking_lin_vel_reward": self.tracking_lin_vel_reward_,
                "tracking_ang_vel_reward": self.tracking_ang_vel_reward_,
                "healthy_reward": self.helthy_reward_,
                "feet_air_time_reward": self.feet_air_time_reward_,
                "torque_cost": self.torque_cost_,
                "action_diff_penalty": self.action_diff_penalty_,
                "lin_vel_z_penalty": self.lin_vel_z_penalty_,
                "xy_angular_velocity_cost": self.xy_angular_velocity_cost_,
                "action_sym": self.action_sym_,
                "acceleration_cost": self.acceleration_cost_,
                "orientation_penalty": self.orientation_penalty_,
                "default_joint_position_cost": self.default_joint_position_cost_,
                'terminated': terminated,
                'truncated:': truncated,
            }

      return obs, reward, terminated, truncated,info


    def save_animation(self, filename="rl_quadruped.gif", fps=10):
      if hasattr(self, 'frames') and self.frames:
          print(f"Saving {len(self.frames)} frames to {filename}")
          self.frames[0].save(
              filename,
              save_all=True,
              append_images=self.frames[1:],
              duration=1000 // fps,
              loop=0
          )

          display(IPImage(filename))
      else:
          print("No frames to save.")


    def _get_obs(self):

        base_pos = self._get_position_data()
        base_vel = self._get_linear_velocity()
        base_orn = self._get_orientation_data()
        base_ang_vel = self._get_roll_pitch_yaw()
        joint_positions = self._get_joint_data()
        joint_velocities = self._get_joint_velocity()
        previous_action = self.prev_action
        reference_vel = self.target_lin_vel
        target_lin_vel = self.target_lin_vel
        target_ang_vel = np.array([0.0])

        feet_contact_force_mag = self.feet_contact_forces
        curr_contact = feet_contact_force_mag > 1.0
        curr_contact = np.array(curr_contact, dtype=np.float32)

        # 3 + 12 + 12 + 4 + 3 + 12
        obs = np.concatenate((
            base_ang_vel ,
            joint_positions ,
            joint_velocities,
            curr_contact,
            target_lin_vel,
            target_ang_vel,
            previous_action), dtype=np.float32)
        return obs

# ===================================== Rewards=================================

    def _compute_reward(self, obs, action):
      # Decompose observation
      base_pos = self._get_position_data()
      base_orn = self._get_orientation_data()
      joint_positions = self._get_joint_data()
      joint_velocities = self._get_joint_velocity()
      lin_vel = self._get_linear_velocity()
      ang_vel = self._get_angular_velocity()
      lin_vel = np.array(lin_vel)
      ang_vel = np.array(ang_vel)



      (tracking_lin_vel_reward,
       tracking_ang_vel_reward,
       lin_vel_z_penalty) = self._tracking_velocity_penalty(lin_vel, ang_vel)

      helthy_reward = self.is_healthy
      feet_air_time_reward = self.feet_air_time_reward


      torque_cost = self.torque_cost(joint_velocities)
      action_diff_penalty = self._action_diff_penalty(action)
      xy_angular_velocity_cost = self.xy_angular_velocity_cost
      joint_limit_cost = self.joint_limit_cost
      acceleration_cost = self.acceleration_cost
      orientation_cost = self.non_flat_base_cost
      default_joint_position_cost = self.default_joint_position_cost
      action_sym = self.action_sym(action)



      Positive_rewards = (tracking_lin_vel_reward *self.reward_weights["linear_vel_tracking"]+
                          tracking_ang_vel_reward * self.reward_weights["angular_vel_tracking"] +
                          helthy_reward * self.reward_weights["healthy"] +
                          feet_air_time_reward * self.reward_weights["feet_airtime"])

      Negative_rewards = (torque_cost * self.cost_weights["torque"] +
                          action_diff_penalty * self.cost_weights["action_rate"] +
                          lin_vel_z_penalty * self.cost_weights["vertical_vel"] +
                          xy_angular_velocity_cost * self.cost_weights["xy_angular_vel"] +
                          action_sym * self.cost_weights["action_sym"] +
                          acceleration_cost * self.cost_weights["joint_acceleration"] +
                          orientation_cost * self.cost_weights["orientation"] +
                          default_joint_position_cost * self.cost_weights["default_joint_position"])


      # reward = max(0,Positive_rewards - Negative_rewards)
      reward= Positive_rewards - Negative_rewards
      # reward = Positive_rewards - self.curriculum_factor * Negative_rewards



      self.tracking_lin_vel_reward_ += tracking_lin_vel_reward *self.reward_weights["linear_vel_tracking"]
      self.tracking_ang_vel_reward_ += tracking_ang_vel_reward * self.reward_weights["angular_vel_tracking"]
      self.helthy_reward_ += helthy_reward * self.reward_weights["healthy"]
      self.feet_air_time_reward_ += feet_air_time_reward * self.reward_weights["feet_airtime"]

      self.torque_cost_ -= torque_cost * self.cost_weights["torque"]
      self.action_diff_penalty_ -= action_diff_penalty * self.cost_weights["action_rate"]
      self.lin_vel_z_penalty_ -= lin_vel_z_penalty * self.cost_weights["vertical_vel"]
      self.xy_angular_velocity_cost_ -= xy_angular_velocity_cost * self.cost_weights["xy_angular_vel"]
      self.action_sym_ -= action_sym * self.cost_weights["action_sym"]
      self.acceleration_cost_ -= acceleration_cost * self.cost_weights["joint_acceleration"]
      self.orientation_penalty_ -= orientation_cost * self.cost_weights["orientation"]
      self.default_joint_position_cost_ -= default_joint_position_cost * self.cost_weights["default_joint_position"]



      self.episode_reward_ += reward

      return reward

    def _height_penalty(self, obs):
      z = self._get_height_data()
      height_penalty = -((z - 0.3)**2)/ 0.05**2
      return height_penalty

    def _pose_penalty(self, obs):
      thigh_indices = np.array([0, 3, 6, 9])
      hip_indices = np.array([1, 4, 7, 10])
      joints_positions = self._get_joint_data()
      pose_error=0
      pose_error = -np.mean(np.square((joints_positions - self.default_joint_pose)))
      # pose_error += -np.sum(np.square(action[hip_indices] - self.default_joint_pose[hip_indices]))

      return pose_error

    def _action_diff_penalty(self, action):
      if not hasattr(self, 'prev_action'):
        self.prev_action = np.zeros_like(action)

      action_diff_penalty = np.sum(np.abs(action - self.prev_action))
      # self.prev_action = action.copy()
      return action_diff_penalty



    def _tracking_velocity_penalty(self, lin_vel, ang_vel):



      # v_x = self.get_projected_vx()
      v_x = lin_vel[0]

      lin_vel_error = np.sum(np.abs(self.target_lin_vel[0] - v_x))

      tracking_lin_vel_reward = np.exp(-lin_vel_error / self.tracking_sigma)
      # tracking_lin_vel_reward = -np.sum(np.abs(self.target_lin_vel - lin_vel[:2]))

      ang_vel_error = np.sum(np.square(self.target_ang_vel - ang_vel[2]))
      tracking_ang_vel_reward = np.exp(-ang_vel_error / self.tracking_sigma)
      # tracking_ang_vel_reward = -np.sum(np.abs(self.target_ang_vel - ang_vel[2]))

      lin_vel_z_penalty = np.square((lin_vel[2]))

      return tracking_lin_vel_reward, tracking_ang_vel_reward, lin_vel_z_penalty

    def get_projected_vx(self):
        roll,pitch,yaw = self._get_roll_pitch_yaw()
        lin_vel = self._get_linear_velocity()
        projected_vx = np.cos(yaw) * lin_vel[0] + np.sin(yaw) * lin_vel[1]
        return projected_vx


    @property
    def feet_air_time_reward(self):
        """Award strides depending on their duration only when the feet makes contact with the ground"""
        feet_contact_force_mag = self.feet_contact_forces
        curr_contact = feet_contact_force_mag > 1.0
        contact_filter = np.logical_or(curr_contact, self._last_contacts)
        self._last_contacts = curr_contact

        # if feet_air_time is > 0 (feet was in the air) and contact_filter detects a contact with the ground
        # then it is the first contact of this stride
        first_contact = (self._feet_air_time > 0.0) * contact_filter
        self._feet_air_time += self.dt

        # Award the feets that have just finished their stride (first step with contact)
        air_time_reward = np.sum((self._feet_air_time) * first_contact)
        # No award if the desired velocity is very low (i.e. robot should remain stationary and feet shouldn't move)
        air_time_reward *= np.linalg.norm(self.target_lin_vel) > 0.1

        # zero-out the air time for the feet that have just made contact (i.e. contact_filter==1)
        self._feet_air_time *= ~contact_filter

        return air_time_reward



    def torque_cost(self,joint_velocities):
        motor_torques = self.get_joint_effort()
        # loss = np.sum(np.abs(motor_torques*joint_velocities))
        loss = np.sum(np.abs(motor_torques))
        return loss


    @property
    def xy_angular_velocity_cost(self):
        return np.sum(np.square(self.data.qvel[3:5]))


    @property
    def joint_limit_cost(self):
        # Penalize the robot for joints exceeding the soft control range
        out_of_range = (self._soft_joint_range[:, 0] - self.data.qpos[7:]).clip(
            min=0.0
        ) + (self.data.qpos[7:] - self._soft_joint_range[:, 1]).clip(min=0.0)
        return np.sum(out_of_range)


    @property
    def acceleration_cost(self):
        return np.sum(np.square(self.data.qacc[6:]))


    @property
    def projected_gravity(self):
        w, x, y, z = self.data.qpos[3:7]
        euler_orientation = np.array(self.euler_from_quaternion(w, x, y, z))
        projected_gravity_not_normalized = (
            np.dot(self._gravity_vector, euler_orientation) * euler_orientation
        )
        if np.linalg.norm(projected_gravity_not_normalized) == 0:
            return projected_gravity_not_normalized
        else:
            return projected_gravity_not_normalized / np.linalg.norm(
                projected_gravity_not_normalized
            )


    # def non_flat_base_cost(self):
    #     # Penalize the robot for not being flat on the ground
    #     return np.sum(np.square(self.projected_gravity[:2]))


    @property
    def default_joint_position_cost2(self):
        return np.sum(np.square(self.data.qpos[7:] - self._default_joint_position))


    @property
    def default_joint_position_cost(self):
        joint_pos = self._get_joint_data()

        soft_joint_limits_low = np.array([
            -0.01, 0.6, -2.1,  # Front left leg
            -0.01, 0.6, -2.1,  # Front right leg
            -0.01, 0.6, -2.1,  # Rear left leg
            -0.01, 0.6, -2.1   # Rear right leg
        ], dtype=np.float32)

        soft_joint_limits_high = np.array([
            0.01, 1.1, -1.5,   # Front left leg
            0.01, 1.1, -1.5,   # Front right leg
            0.01, 1.1, -1.5,   # Rear left leg
            0.01, 1.1, -1.5    # Rear right leg
        ], dtype=np.float32)

        lower_violation = np.maximum(soft_joint_limits_low - joint_pos, 0)
        upper_violation = np.maximum(joint_pos - soft_joint_limits_high, 0)

        total_violation = lower_violation + upper_violation

        # Square the violations to penalize larger ones more
        return np.sum(np.square(total_violation))


    @property
    def non_flat_base_cost(self):

        roll, pitch, _ = self._get_roll_pitch_yaw()

        return np.square(roll) + np.square(pitch)


    def action_sym2(self, action):

        diagonal_pairs = [
              (1,7),
              (4,10)]

        jointpositions = self._get_joint_data()
        loss = sum((jointpositions[i] - jointpositions[j]) ** 2 for i, j in diagonal_pairs)

        return loss

    def action_sym(self, action):
    # Diagonal pairs expected to move in-phase (same value)
        in_phase_pairs = [(1, 7), (4, 10)]

        out_of_phase_pairs = [(1, 4), (7, 10)]


        jointpositions = self._get_joint_data()  # array of normalized joint angles in [0, 1]

        loss_in = sum((jointpositions[i] - jointpositions[j]) ** 2 for i, j in in_phase_pairs)

        loss_out = sum((jointpositions[i] + jointpositions[j] - 1.5) ** 2 for i, j in out_of_phase_pairs)
        total_loss = loss_in + loss_out

        return total_loss

# ======================================================================



    @property
    def is_healthy(self):
        x,y,z = self._get_position_data()
        roll, pitch, _ = self._get_roll_pitch_yaw()

        min_z, max_z = self._healthy_z_range
        is_healthy = min_z <= z <= max_z

        min_roll, max_roll = self._healthy_roll_range
        is_healthy = is_healthy and min_roll <= roll <= max_roll

        min_pitch, max_pitch = self._healthy_pitch_range
        is_healthy = is_healthy and min_pitch <= pitch <= max_pitch

        return is_healthy


    @property
    def feet_contact_forces(self):
        feet_contact_forces = self.data.cfrc_ext[self._cfrc_ext_feet_indices].copy()
        return np.linalg.norm(feet_contact_forces, axis=1)




    @staticmethod
    def euler_from_quaternion(w, x, y, z):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return roll_x, pitch_y, yaw_z


# ======================================================================

    @property
    def curriculum_factor(self):
        return self._curriculum_base**0.997


    def _check_termination(self, obs):

        base_pos = self._get_position_data()
        base_orn = self._get_orientation_data()
        joint_positions = self._get_joint_data()
        joint_velocities = self._get_joint_velocity()


        x, y, z = base_pos

        # 1. Height check - robot fell over
        if z < 0.15:
            if self.episode_counter % self.log_episode_count == 0:
                print("height fell")
            return True


        if x >= self.goal_distance:
          self.reached_target = True
          if self.episode_counter % self.log_episode_count == 0:
              print("goal reached")
          return True

        # 2. Excessive tilt - robot is too tilted to recover
        roll, pitch ,_= self._get_roll_pitch_yaw()
        max_tilt = np.pi/6  # 60 degrees
        if abs(roll) > max_tilt or abs(pitch) > max_tilt:
            if self.episode_counter % self.log_episode_count == 0:
                print("too tilted")
            return True

        # 3. Lateral drift - robot moved too far sideways
        max_lateral_drift = 2.0  # meters
        if abs(y) > max_lateral_drift:
            if self.episode_counter % self.log_episode_count == 0:
                print("lateral drift")
            return True



        return False
    def create_animation(self, filename="go2_walk.gif", fps=30):
        if not self.simulation_data['images']:
            print("No images captured.")
            return
        self.simulation_data['images'][0].save(
            filename, save_all=True, append_images=self.simulation_data['images'][1:],
            duration=1000 // fps, loop=0
        )
        print(f"Animation saved to {filename}")



    def _get_position_data(self):
        return self.data.qpos[0:3].copy()

    def _get_orientation_data(self):
        return self.data.qpos[3:7].copy()

    def _get_joint_data(self):
        return self.data.qpos[7:19].copy()

    def _get_height_data(self):
        return self._get_position_data()[-1]



    def _get_roll_pitch_yaw(self):
        quat = self._get_orientation_data()
        w, x, y, z = quat
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(2*(w*y - z*x))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return roll, pitch, yaw



    def get_joint_effort(self):
        return self.data.ctrl[:12].copy()

    def _get_velocity_data(self):
        return self.data.qvel.copy()

    def _get_linear_velocity(self):
        return self._get_velocity_data()[0:3]

    def _get_angular_velocity(self):
        return self._get_velocity_data()[3:6]

    def _get_joint_velocity(self):
        return self._get_velocity_data()[6:]


    def _get_joint_names(self):

        print("Joint Names and Indices:")
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            print(f"  Index {i}: {joint_name}")


    def set_joint_positions(self, joint_angles):
        self.data.ctrl[:12] = joint_angles


    def capture_frame(self):
        self.cam.azimuth = 120.0
        self.cam.lookat[:] = self._get_position_data()
        self.renderer.update_scene(self.data,self.cam)
        img = self.renderer.render()
        return Image.fromarray(img)

    def render(self, mode='human'):
      if not hasattr(self, 'frames'):
          self.frames = []

      img = self.capture_frame()
      self.frames.append(img)

    def close(self):
        return super().close()






def make_env(rank):
    def _init():
        env = QuadrupedEnv()
        return env
    return _init

def cleanup_env(envs):
    envs.close()



reward_weights = {
    "linear_vel_tracking": 3,  # Was 1.0
    "angular_vel_tracking":0.5,
    "healthy": 2.5,  # was 0.05
    "feet_airtime": 15,
}
cost_weights = {
    "torque": 0.3,
    "vertical_vel": 0.0,  # Was 1.0
    "xy_angular_vel": 0.05*0,  # Was 0.05
    "action_rate": 0.6,
    "action_sym": 0.3,
    "joint_velocity": 0.01*0,
    "joint_acceleration": 2.5e-7*0,
    "orientation": 1.0*0,
    "collision": 1.0*0,
    "default_joint_position": 0.0
}
retrain = False



single_env = QuadrupedEnv()
single_env.set_weights(reward_weights, cost_weights)
obs, info = single_env.reset()
# name = "./Models/Mujoco v 3 2025-08-17_14-18-45" #walking
# name = "./Models/Running_Model_2025-08-18_10-47-08" #trot
name = "./Models/Running_Model_2025-08-19_20-32-19" #running
viewer_model = PPO.load(name, env=single_env)

actions = []
joint_positions = []


print("Launching MuJoCo viewer...")
with mujoco.viewer.launch_passive(single_env.model, single_env.data) as viewer:
    print("MuJoCo viewer launched successfully!")
    step_count = 0
    
    while viewer.is_running() and step_count < 250:
        step_start = time.time()
        
        action, _ = viewer_model.predict(obs, deterministic=True)
        # action = single_env._default_joint_position.copy()  # Use the default joint position as a placeholder action
        actions.append(action)
        print(f"Step {step_count}: Action = {action}") 
        obs, reward, done, trunc, info = single_env.step(action)
        joint_positions.append(single_env._get_joint_data().copy())
        single_env.render()
            
        if single_env.model.nbody > 1:
            robot_body_id = 1
        else:
            robot_body_id = 0
            
        robot_pos = single_env.data.body(robot_body_id).xpos

        # Update the camera's lookat position to the robot's position
        viewer.cam.lookat[:] = robot_pos

        viewer.sync()
        step_count += 1
        
        # Control simulation speed
        time_until_next_step = single_env.dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
# Save the actions and joint positions to one file
# np.savez("actions_and_joint_positions.npz", actions=actions, joint_positions=joint_positions)
single_env.save_animation("quadruped_running.gif", fps=25)
