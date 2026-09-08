"""Optional TD3 policy components matching the Tomas TurtleBot3 agent."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import TD3Policy


def initialize_tomas_weights(module: nn.Module) -> None:
    """Apply the initialization used by the Tomas actor and critic."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.01)


class TomasQNetwork(nn.Module):
    """One Tomas critic head with separate state and action projections."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        branch_size = hidden_size // 2
        self.state_layer = nn.Linear(state_dim, branch_size)
        self.action_layer = nn.Linear(action_dim, branch_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        state_features = torch.relu(self.state_layer(states))
        action_features = torch.relu(self.action_layer(actions))
        joint_features = torch.cat((state_features, action_features), dim=1)
        hidden = torch.relu(self.hidden_layer(joint_features))
        return self.output_layer(hidden)


class TomasContinuousCritic(BaseModel):
    """Twin critic with the exact state/action split used by Tomas TD3."""

    def __init__(
        self,
        observation_space,
        action_space,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        del activation_fn
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        if not net_arch:
            raise ValueError("Tomas critic requires a hidden layer size")
        hidden_size = int(net_arch[0])
        if hidden_size % 2:
            raise ValueError("Tomas critic hidden size must be even")

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.features_dim = features_dim
        action_dim = int(action_space.shape[0])

        self.q_networks: list[TomasQNetwork] = []
        for index in range(n_critics):
            q_network = TomasQNetwork(features_dim, action_dim, hidden_size)
            self.add_module(f"qf{index}", q_network)
            self.q_networks.append(q_network)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        with torch.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(observations, self.features_extractor)
        return tuple(q_network(features, actions) for q_network in self.q_networks)

    def q1_forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            features = self.extract_features(observations, self.features_extractor)
        return self.q_networks[0](features, actions)


class ConfigurableTD3Policy(TD3Policy):
    """SB3 TD3 policy with independently selectable Tomas critic and init."""

    def __init__(
        self,
        *args: Any,
        critic_architecture: str = "sb3",
        xavier_init: bool = False,
        **kwargs: Any,
    ):
        self.critic_architecture = critic_architecture
        self.xavier_init = xavier_init
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor=None):
        actor = super().make_actor(features_extractor)
        if self.xavier_init:
            actor.apply(initialize_tomas_weights)
        return actor

    def make_critic(self, features_extractor=None):
        if self.critic_architecture == "sb3":
            critic = super().make_critic(features_extractor)
        elif self.critic_architecture == "tomas":
            critic_kwargs = self._update_features_extractor(
                self.critic_kwargs, features_extractor
            )
            critic = TomasContinuousCritic(**critic_kwargs).to(self.device)
        else:
            raise ValueError(
                f"Unknown critic architecture: {self.critic_architecture}"
            )

        if self.xavier_init:
            critic.apply(initialize_tomas_weights)
        return critic

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            critic_architecture=self.critic_architecture,
            xavier_init=self.xavier_init,
        )
        return data
