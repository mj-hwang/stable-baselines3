import sys
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import BaseBuffer, MixedReplayBuffer, BalancedMixedReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update, safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
# from stable_baselines3.maple.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3.seed.policies import MlpMixedPolicy, SEEDMixedPolicy

SelfSEED = TypeVar("SelfSEED", bound="SEED")


class SEED(OffPolicyAlgorithm):
    """
    MAPLE + TAMER with Oracle Feedback
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
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
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
        "MlpMixedPolicy": MlpMixedPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[SEEDMixedPolicy]],
        env: Union[GymEnv, str],
        trained_model,
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
        replay_buffer_class: Optional[Type[BaseBuffer]] = BalancedMixedReplayBuffer,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
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
        q_val_threshold: float = 0.999,
        delta_q_val_threshold: float = 0.0,
        rl_threshold: float = 0.0,
        percent_feedback: float = 1.0,
        use_bc: bool = False,
        use_supervised_q: bool = False,
        use_env_feedback: bool = False,
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
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box),
            support_multi_env=True,
        )

        self.num_skill_timesteps = 0

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

        self.trained_model = trained_model
        self.q_val_threshold = q_val_threshold
        self.delta_q_val_threshold = delta_q_val_threshold
        self.rl_threshold = rl_threshold
        self.percent_feedback = percent_feedback
        self.total_feedback = 0
        self.use_bc = use_bc
        self.use_supervised_q = use_supervised_q
        self.use_env_feedback = use_env_feedback

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.replay_buffer = self.replay_buffer_class(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            n_envs=self.n_envs,
            optimize_memory_usage=self.optimize_memory_usage,
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
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
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
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        self.human_critic = self.policy.human_critic
        self.human_critic_target = self.policy.human_critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.human_critic.optimizer]
        if self.ent_coef_s_optimizer is not None:
            optimizers += [self.ent_coef_s_optimizer]
        if self.ent_coef_p_optimizer is not None:
            optimizers += [self.ent_coef_p_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_s_losses, ent_coefs_s = [], []
        ent_coef_p_losses, ent_coefs_p = [], []
        actor_losses, critic_losses, human_critic_losses = [], [], []

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
                # Select action according to policy
                next_actions, next_log_prob_s, next_log_prob_p = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term 
                next_q_values = next_q_values - ent_coef_s * next_log_prob_s.reshape(-1, 1) - ent_coef_p * next_log_prob_p.reshape(-1, 1)
                # next_q_values = next_q_values - ent_coef_p * next_log_prob_p.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                # target human q vals
                target_human_q_values = replay_data.human_rewards

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Get current Q-values estimates for human critic network
            # using action from the replay buffer
            current_human_q_values = self.human_critic(replay_data.observations, replay_data.actions)

            # Compute human critic loss
            human_critic_loss = 0.5 * sum([F.mse_loss(current_human_q, target_human_q_values) for current_human_q in current_human_q_values])
            human_critic_losses.append(human_critic_loss.item())

            # Optimize the human critic
            self.human_critic.optimizer.zero_grad()
            human_critic_loss.backward()
            self.human_critic.optimizer.step()

            if self.use_bc:
                with th.no_grad():
                    teacher_actions, _ = self.trained_model.actor.action_log_prob(replay_data.observations)
                mseloss = th.nn.MSELoss()
                actor_loss = mseloss(actions_pi, teacher_actions)
            else:
                # Compute actor loss
                # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
                # Min over all critic networks
                q_values_pi_critic = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
                q_values_pi_human = th.cat(self.human_critic(replay_data.observations, actions_pi), dim=1)
                q_values_pi = (
                    self.rl_threshold * q_values_pi_critic
                    + (1 - self.rl_threshold) * q_values_pi_human
                )
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef_s * log_prob_s + ent_coef_p * log_prob_p - min_qf_pi).mean()
                # actor_loss = (ent_coef_p * log_prob_p - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.human_critic.parameters(), self.human_critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.human_critic_batch_norm_stats, self.human_critic_batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef_s", np.mean(ent_coefs_s))
        self.logger.record("train/ent_coef_p", np.mean(ent_coefs_p))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/human_critic_loss", np.mean(human_critic_losses))
        self.logger.record("train/rl_threshold", self.rl_threshold)
        self.logger.record("train/q_val_threshold", self.q_val_threshold)
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
        self.logger.record("time/total_skill_timesteps", self.num_skill_timesteps)
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target", "human_critic", "human_critic_target", "trained_model"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer", "human_critic.optimizer"]
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
        replay_buffer: MixedReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``MixedReplayBuffer``.

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
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            if self.use_supervised_q:
                with th.no_grad():
                    student_q_values = self.trained_model.critic(
                        th.from_numpy(self._last_obs).to(self.device),
                        th.from_numpy(actions).to(self.device),
                    )
                    student_q_values, _ = th.min(th.cat(student_q_values, dim=1), dim=1, keepdim=True)
                    simulated_human_rewards = student_q_values.cpu()
            elif self.use_env_feedback:
                simulated_human_rewards = np.array([env.envs[i].human_reward(actions[i]) for i in range(env.num_envs)])
            else:
                with th.no_grad():
                    student_q_values = self.trained_model.critic(
                        th.from_numpy(self._last_obs).to(self.device),
                        th.from_numpy(actions).to(self.device),
                    )
                    student_q_values, _ = th.min(th.cat(student_q_values, dim=1), dim=1, keepdim=True)

                    teacher_actions, _ = self.trained_model.predict(self._last_obs)
                    teacher_q_values = self.trained_model.critic(
                        th.from_numpy(self._last_obs).to(self.device),
                        th.from_numpy(teacher_actions).to(self.device),
                    )
                    teacher_q_values, _ = th.min(th.cat(teacher_q_values, dim=1), dim=1, keepdim=True)

                    simulated_human_rewards = self.q_val_threshold * teacher_q_values < student_q_values
                    simulated_human_rewards = simulated_human_rewards.float()
                    simulated_human_rewards = simulated_human_rewards * 2 - 1
                    simulated_human_rewards = simulated_human_rewards.cpu()
                
                self.q_val_threshold = min(1.0, self.q_val_threshold + self.delta_q_val_threshold)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += sum([info["num_timesteps"] for info in infos])
            self.num_skill_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, simulated_human_rewards, dones, infos)

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
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _store_transition(
        self,
        replay_buffer: MixedReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        human_reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            human_reward,
            dones,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_
