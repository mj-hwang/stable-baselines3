import os
import sys
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import BaseBuffer, HumanReplayBuffer, BalancedHumanReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update, safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
# from stable_baselines3.maple.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3.seed.policies import MlpHumanPolicy, MlpDiscreteHumanPolicy, SEEDHumanPolicy, SEEDDiscreteHumanPolicy

SelfSEED = TypeVar("SelfSEEDHuman", bound="SEEDHuman")


class SEEDHuman(OffPolicyAlgorithm):
    """
    MAPLE + TAMER with Human Feedback
    Off-Policy Maximum Entropy Deep Reinforcement Learning with Hierarichal Actors and Primitives.
    This implementation borrows code from Stable Baselines 3.
    MAPLE Paper: https://arxiv.org/abs/2110.03655
    SAC Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpHumanPolicy": MlpHumanPolicy,
        "MlpDiscreteHumanPolicy": MlpDiscreteHumanPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[SEEDHumanPolicy]],
        env: Union[GymEnv, str],
        action_dim_s: int = 0,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[BaseBuffer]] = BalancedHumanReplayBuffer,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        ent_coef_s: Union[str, float] = "auto",
        ent_coef_p: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy_s: Union[str, float] = "auto",
        target_entropy_p: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        log_cumulative_reward: bool = True,
        save_freq: int = -1,
    ):

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=False,
            supported_action_spaces=(spaces.Box),
            support_multi_env=True,
        )

        print(self.policy_class.__name__)
        self.use_discrete = self.policy_class.__name__ == "SEEDDiscreteHumanPolicy"

        self.num_skill_timesteps = 0
        self.num_feedbacks = 0
        self.num_ll_steps = 0
        self.num_hl_steps = 0
        self.save_freq = save_freq

        self.log_cumulative_reward = log_cumulative_reward
        self.cumulative_reward = 0

        self.loss_weight = th.Tensor([1.0, 10.0])

        self.target_entropy_s = target_entropy_s
        self.target_entropy_p = target_entropy_p
        self.log_ent_coef_s = None  # type: Optional[th.Tensor]
        self.log_ent_coef_p = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef_s = ent_coef_s
        self.ent_coef_p = ent_coef_p
        self.target_update_interval = target_update_interval
        self.ent_coef_s_optimizer = None
        self.ent_coef_p_optimizer = None
        
        self.action_dim = self.env.action_space.low.size
        self.action_dim_s = action_dim_s
        self.action_dim_p = self.action_dim - self.action_dim_s

        if _init_setup_model:
            self._setup_model()

    def weighted_mse_loss(self, input, target, weight):
        return (weight * (input - target) ** 2).mean()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.replay_buffer = self.replay_buffer_class(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            n_envs=self.n_envs,
            **self.replay_buffer_kwargs,
        )

        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            self.action_dim_s,
            self.action_dim_p,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()

        self._create_aliases()
        # Running mean and running var
        self.human_critic_batch_norm_stats = get_parameters_by_name(self.human_critic, ["running_"])
        self.human_critic_batch_norm_stats_target = get_parameters_by_name(self.human_critic_target, ["running_"])
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy_s == "auto":
            # automatically set target entropy if needed
            self.target_entropy_s = -np.prod(self.action_dim_s).astype(np.float32)
            # # since we use one-hot encoding, we scale accordingly
            # self.target_entropy_s = np.log(self.action_dim_s) * 0.75
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy_s = float(self.target_entropy_s)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef_s, str) and self.ent_coef_s.startswith("auto"):
            # Default initial value of ent_coef_s when learned
            init_value = 1.0
            if "_" in self.ent_coef_s:
                init_value = float(self.ent_coef_s.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef_s = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_s_optimizer = th.optim.Adam([self.log_ent_coef_s], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_s_tensor = th.tensor(float(self.ent_coef_s), device=self.device)

        if self.target_entropy_p == "auto":
            # automatically set target entropy if needed
            self.target_entropy_p = -np.prod(self.action_dim_p).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy_p = float(self.target_entropy_p)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef_p, str) and self.ent_coef_p.startswith("auto"):
            # Default initial value of ent_coef_p when learned
            init_value = 1.0
            if "_" in self.ent_coef_p:
                init_value = float(self.ent_coef_p.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef_p = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_p_optimizer = th.optim.Adam([self.log_ent_coef_p], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_p_tensor = th.tensor(float(self.ent_coef_p), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.human_critic = self.policy.human_critic
        self.human_critic_target = self.policy.human_critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        print("train being called")
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.human_critic.optimizer]
        if self.ent_coef_s_optimizer is not None:
            optimizers += [self.ent_coef_s_optimizer]
        if self.ent_coef_p_optimizer is not None:
            optimizers += [self.ent_coef_p_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_s_losses, ent_coefs_s = [], []
        ent_coef_p_losses, ent_coefs_p = [], []
        actor_losses, human_critic_losses = [], []
        ent_s_losses, ent_p_losses, actor_q_losses, bc_losses = [], [], [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob_s, log_prob_p = self.actor.action_log_prob(replay_data.observations)
            log_prob_s = log_prob_s.reshape(-1, 1)
            log_prob_p = log_prob_p.reshape(-1, 1)

            ent_coef_s_loss = None
            if self.ent_coef_s_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef_s = th.exp(self.log_ent_coef_s.detach())
                ent_coef_s_loss = -(self.log_ent_coef_s * (log_prob_s + self.target_entropy_s).detach()).mean()
                ent_coef_s_losses.append(ent_coef_s_loss.item())
            else:
                ent_coef_s = self.ent_coef_s_tensor

            ent_coefs_s.append(ent_coef_s.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_s_loss is not None:
                self.ent_coef_s_optimizer.zero_grad()
                ent_coef_s_loss.backward()
                self.ent_coef_s_optimizer.step()

            ent_coef_p_loss = None
            if self.ent_coef_p_optimizer is not None:
                ent_coef_p = th.exp(self.log_ent_coef_p.detach())
                ent_coef_p_loss = -(self.log_ent_coef_p * (log_prob_p + self.target_entropy_p).detach()).mean()
                ent_coef_p_losses.append(ent_coef_p_loss.item())
            else:
                ent_coef_p = self.ent_coef_p_tensor

            ent_coefs_p.append(ent_coef_p.item())

            if ent_coef_p_loss is not None:
                self.ent_coef_p_optimizer.zero_grad()
                ent_coef_p_loss.backward()
                self.ent_coef_p_optimizer.step()

            with th.no_grad():
                # target human q vals
                target_human_q_values = replay_data.human_rewards

            # Get current Q-values estimates for human critic network
            # using action from the replay buffer
            current_human_q_values = self.human_critic(replay_data.observations, replay_data.actions)

            if not self.use_discrete:
                # Compute human critic loss\
                # good_action_index = target_human_q_values == 1
                # weights = good_action_index.float() * 19 + 1
                # human_critic_loss = 0.5 * sum([self.weighted_mse_loss(current_human_q, target_human_q_values, weights) for current_human_q in current_human_q_values])
                human_critic_loss = 0.5 * sum([F.mse_loss(current_human_q, target_human_q_values) for current_human_q in current_human_q_values])
            else:
                # convert human rewards to classes ({-1, 1} -> {0, 1})
                target_human_q_values_labels = ((target_human_q_values.squeeze(-1) + 1) / 2).to(dtype=th.long)
                human_critic_loss = F.cross_entropy(current_human_q_values, target_human_q_values_labels, weight=self.loss_weight)
            
            human_critic_losses.append(human_critic_loss.item())

            # Optimize the human critic
            self.human_critic.optimizer.zero_grad()
            human_critic_loss.backward()
            self.human_critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            if not self.use_discrete:
                q_values_pi = th.cat(self.human_critic(replay_data.observations, actions_pi), dim=1)
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            else:
                min_qf_pi = self.human_critic._predict(replay_data.observations, actions_pi).unsqueeze(-1)
            
            # good_action_index = good_action_index.squeeze(-1)
            # good_actions = replay_data.actions[good_action_index]
            # current_actions = actions_pi[good_action_index]
            # good_action_same_skill_index = current_actions[:, :self.action_dim_s].argmax(dim=1) == good_actions[:, :self.action_dim_s].argmax(dim=1)
            
            # bc_loss = F.mse_loss(current_actions[good_action_same_skill_index, self.action_dim_s:], 
            #                      good_actions[good_action_same_skill_index, self.action_dim_s:])


            ent_s_loss = (ent_coef_s * log_prob_s).mean()
            ent_p_loss = (ent_coef_p * log_prob_p).mean()
            actor_q_loss = (- min_qf_pi).mean()
            actor_loss = (ent_coef_s * log_prob_s + ent_coef_p * log_prob_p - min_qf_pi).mean()
            # actor_loss = ent_s_loss + ent_p_loss + actor_q_loss
            
            # print("ent_s_loss:", (ent_coef_s * log_prob_s).mean())
            # print("ent_p_loss:", (ent_coef_p * log_prob_p).mean())
            # print("actor_q_loss:", actor_q_loss)
            # print("BC loss mean", bc_loss.mean())
            
            ent_s_losses.append(ent_s_loss.item())
            ent_p_losses.append(ent_p_loss.item())
            actor_q_losses.append(actor_q_loss.item())
            # bc_losses.append(bc_loss.item())
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.human_critic.parameters(), self.human_critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.human_critic_batch_norm_stats, self.human_critic_batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef_s", np.mean(ent_coefs_s))
        self.logger.record("train/ent_coef_p", np.mean(ent_coefs_p))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/ent_s_loss", np.mean(ent_s_losses))
        self.logger.record("train/ent_p_loss", np.mean(ent_p_losses))
        self.logger.record("train/actor_q_loss", np.mean(actor_q_losses))
        # self.logger.record("train/bc_loss", np.mean(bc_losses))
        self.logger.record("train/human_critic_loss", np.mean(human_critic_losses))
        if len(ent_coef_s_losses) > 0:
            self.logger.record("train/ent_coef_s_loss", np.mean(ent_coef_s_losses))
        if len(ent_coef_p_losses) > 0:
            self.logger.record("train/ent_coef_p_loss", np.mean(ent_coef_p_losses))

    def learn(
        self: SelfSEED,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SEED",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSEED:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
    
    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps)
        self.logger.record("time/low_level_action_steps", self.num_ll_steps)
        self.logger.record("time/high_level_action_steps", self.num_hl_steps)
        self.logger.record("time/num_feedbacks", self.num_feedbacks)
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())
        if self.log_cumulative_reward:
            self.logger.record("time/cumulative_reward", self.cumulative_reward)

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "human_critic", "human_critic_target", "trained_model"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "human_critic.optimizer"]
        saved_pytorch_variables = []
        if self.ent_coef_s_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef_s"]
            state_dicts.append("ent_coef_s_optimizer")
        else:
            saved_pytorch_variables.append("ent_coef_s_tensor")
        if self.ent_coef_p_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef_p"]
            state_dicts.append("ent_coef_p_optimizer")
        else:
            saved_pytorch_variables.append("ent_coef_p_tensor")
        return state_dicts, saved_pytorch_variables
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: HumanReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``HumanReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            print("osb:", self._last_obs)
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # np array
            # will be replace with actual feedbacks
            human_rewards = np.array([env.envs[i].human_reward(actions[i])[0] for i in range(env.num_envs)])

            self.num_feedbacks += env.num_envs

            new_obs, rewards, dones, infos = env.step(actions)
            self.num_ll_steps += sum([info["num_ll_steps"] for info in infos]) # can be zero
            self.num_hl_steps += sum([info["num_hl_steps"] for info in infos])
            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, human_rewards, dones, infos, new_obs=new_obs)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

            if self.save_freq != -1 and self.num_feedbacks % self.save_freq == 0:
                self._dump_logs()
                print(f"Saving the model; Current Num Feedbacks: {self.num_feedbacks}")
                self.save(os.path.join(self.tensorboard_log, f"model_num_feedbacks_{self.num_feedbacks}"))
                self.save_replay_buffer(os.path.join(self.tensorboard_log, f"replaybuffer_num_feedbacks_{self.num_feedbacks}"))

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _store_transition(
        self,
        replay_buffer: HumanReplayBuffer,
        buffer_action: np.ndarray,
        human_reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
        new_obs: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
    ) -> None:
        # Store only the unnormalized version
        if self._vec_normalize_env is None:
            self._last_original_obs = self._last_obs

        replay_buffer.add(
            self._last_original_obs,
            buffer_action,
            human_reward,
            dones,
            infos,
        )
        
        if new_obs is not None:
            self._last_obs = new_obs
            # Save the unnormalized observation
            if self._vec_normalize_env is not None:
                self._last_original_obs = new_obs_

    def load_replay_buffer(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        truncate_last_traj: bool = True,
    ) -> None:
        """
        Load a replay buffer from a pickle file.

        :param path: Path to the pickled replay buffer.
        :param truncate_last_traj: When using ``HerReplayBuffer`` with online sampling:
            If set to ``True``, we assume that the last trajectory in the replay buffer was finished
            (and truncate it).
            If set to ``False``, we assume that we continue the same trajectory (same episode).
        """
        self.replay_buffer = load_from_pkl(path, self.verbose)
        # assert isinstance(self.replay_buffer, ReplayBuffer), "The replay buffer must inherit from ReplayBuffer class"

        # Backward compatibility with SB3 < 2.1.0 replay buffer
        # Keep old behavior: do not handle timeout termination separately
        if not hasattr(self.replay_buffer, "handle_timeout_termination"):  # pragma: no cover
            self.replay_buffer.handle_timeout_termination = False
            self.replay_buffer.timeouts = np.zeros_like(self.replay_buffer.dones)