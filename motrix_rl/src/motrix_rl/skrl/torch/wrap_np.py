# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Any, Tuple

import gymnasium
import numpy as np
import torch
from skrl.envs.torch import Wrapper as SkrlWrapper

from motrix_envs.np.env import NpEnv
from motrix_envs.np.renderer import NpRenderer


class SkrlNpWrapper(SkrlWrapper):
    """
    Wrap the numpy-based environment to be compatible with skrl (PyTorch).

    Optimized for GPU training with:
    - Pre-allocated pinned-memory CPU buffers for async CPU→GPU transfers
    - ``torch.from_numpy`` (zero-copy on CPU) + ``non_blocking`` ``.to()``
    - Pre-allocated pinned action buffer for async GPU→CPU transfers

    When device is CPU, falls back to simple ``torch.from_numpy`` without pinning.
    """

    _env: NpEnv
    _renderer: NpRenderer = None

    # Pre-allocated pinned-memory buffers (GPU path only)
    _pin_obs: torch.Tensor = None
    _pin_reward: torch.Tensor = None
    _pin_terminated: torch.Tensor = None
    _pin_truncated: torch.Tensor = None
    _pin_actions: torch.Tensor = None

    def __init__(self, env: NpEnv, enable_render: bool = False):
        super().__init__(env)
        if enable_render:
            self._renderer = NpRenderer(env)

        self._use_cuda = self.device.type == "cuda"
        if self._use_cuda:
            self._init_pinned_buffers()

    # -- internal helpers -----------------------------------------------------

    def _init_pinned_buffers(self) -> None:
        """Pre-allocate page-locked CPU tensors sized to ``num_envs``."""
        n = self._env.num_envs
        obs_dim = self._env.observation_space.shape[0]
        act_dim = self._env.action_space.shape[0]

        self._pin_obs = torch.empty((n, obs_dim), dtype=torch.float32).pin_memory()
        self._pin_reward = torch.empty((n, 1), dtype=torch.float32).pin_memory()
        self._pin_terminated = torch.empty((n, 1), dtype=torch.bool).pin_memory()
        self._pin_truncated = torch.empty((n, 1), dtype=torch.bool).pin_memory()
        self._pin_actions = torch.empty((n, act_dim), dtype=torch.float32).pin_memory()

    def _np_to_gpu(self, buf: torch.Tensor, arr: np.ndarray) -> torch.Tensor:
        """Copy *arr* into the pre-allocated pinned *buf* and async-transfer to GPU."""
        buf.copy_(torch.from_numpy(arr))  # CPU memcpy into pinned buffer
        return buf.to(self.device, non_blocking=True)

    def _np_to_device(self, arr: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """CPU-only fallback: zero-copy wrap + device copy."""
        return torch.from_numpy(arr).to(device=self.device, dtype=dtype)

    # -- public API -----------------------------------------------------------

    def reset(self) -> Tuple[torch.Tensor, Any]:
        state = self._env.init_state()
        if self._use_cuda:
            obs = self._np_to_gpu(self._pin_obs, state.obs)
        else:
            obs = self._np_to_device(state.obs)
        return obs, state.info

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Any,
    ]:
        # GPU→CPU: async copy into pinned buffer, then get numpy view
        if self._use_cuda:
            self._pin_actions.copy_(actions, non_blocking=True)
            # Synchronize to ensure the copy is complete before numpy() reads the data
            torch.cuda.current_stream().synchronize()
            np_actions = self._pin_actions.numpy()
        else:
            np_actions = actions.detach().numpy()

        state = self._env.step(np_actions)

        if self._use_cuda:
            obs = self._np_to_gpu(self._pin_obs, state.obs)
            reward = self._np_to_gpu(self._pin_reward, state.reward.reshape(-1, 1))
            terminated = self._np_to_gpu(self._pin_terminated, state.terminated.reshape(-1, 1))
            truncated = self._np_to_gpu(self._pin_truncated, state.truncated.reshape(-1, 1))
        else:
            obs = self._np_to_device(state.obs)
            reward = self._np_to_device(state.reward.reshape(-1, 1))
            terminated = torch.from_numpy(state.terminated.reshape(-1, 1)).to(device=self.device)
            truncated = torch.from_numpy(state.truncated.reshape(-1, 1)).to(device=self.device)

        return obs, reward, terminated, truncated, state.info

    def render(self, *args, **kwargs) -> Any:
        if self._renderer:
            self._renderer.render()

    def close(self) -> None:
        pass

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def observation_space(self) -> gymnasium.Space:
        return self._env.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        return self._env.action_space
