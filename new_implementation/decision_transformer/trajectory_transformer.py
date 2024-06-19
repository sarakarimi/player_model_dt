from abc import abstractmethod
from typing import Tuple, Union
import einops
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from gymnasium.spaces import Box, Dict
from torchtyping import TensorType as TT
from transformer_lens import HookedTransformer, HookedTransformerConfig

from new_implementation.configs import EnvironmentConfig, TransformerModelConfig


class TrajectoryTransformer(nn.Module):
    """
    Base Class for trajectory modelling transformers including:
        - Decision Transformer (offline, RTG, (R,s,a))
        - Online Transformer (online, reward, (s,a,r) or (s,a))
    """

    def __init__(
            self,
            transformer_config: TransformerModelConfig,
            environment_config: EnvironmentConfig,
    ):
        super().__init__()

        self.transformer_config = transformer_config
        self.environment_config = environment_config

        # Why is this in a sequential? Need to get rid of it at some
        # point when I don't care about loading older models.
        self.action_embedding = nn.Sequential(
            nn.Embedding(
                environment_config.action_space.n + 1,
                self.transformer_config.d_model,
            )
        )
        self.time_embedding = self.initialize_time_embedding()
        self.state_embedding = self.initialize_state_embedding()

        # Initialize weights
        nn.init.normal_(
            self.action_embedding[0].weight,
            mean=0.0,
            std=1
                / (
                        (environment_config.action_space.n + 1 + 1)
                        * self.transformer_config.d_model
                ),
        )

        self.transformer = self.initialize_easy_transformer()

        self.action_predictor = nn.Linear(
            self.transformer_config.d_model * 2, environment_config.action_space.n
        )
        self.initialize_state_predictor()

        self.initialize_weights()

    def initialize_weights(self):
        """
        TransformerLens is weird so we have to use the module path
        and can't just rely on the module instance as we do would
        be the default approach in pytorch.
        """
        self.apply(self._init_weights_classic)

        for name, param in self.named_parameters():
            if "W_" in name:
                nn.init.normal_(param, std=0.02)

    def _init_weights_classic(self, module):
        """
        Use Min GPT Method.
        https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L163

        Will need to check that this works with the transformer_lens library.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif (
                "PosEmbedTokens" in module._get_name()
        ):  # transformer lens components
            for param in module.parameters():
                torch.nn.init.normal_(param, mean=0.0, std=0.02)

    def get_time_embedding(self, timesteps):
        assert (
                timesteps.max() <= self.environment_config.max_steps
        ), "timesteps must be less than max_timesteps"

        block_size = timesteps.shape[1]
        timesteps = rearrange(
            timesteps, "batch block time-> (batch block) time"
        )
        time_embeddings = self.time_embedding(timesteps)
        if self.transformer_config.time_embedding_type != "linear":
            time_embeddings = time_embeddings.squeeze(-2)
        time_embeddings = rearrange(
            time_embeddings,
            "(batch block) n_embd -> batch block n_embd",
            block=block_size,
        )
        return time_embeddings

    def get_state_embedding(self, states):
        # embed states and recast back to (batch, block_size, n_embd)
        block_size = states.shape[1]
        if self.transformer_config.state_embedding_type.lower() in [
            "cnn",
            "vit",
        ]:
            states = rearrange(
                states,
                "batch block height width channel -> (batch block) height width channel",
            )
            state_embeddings = self.state_embedding(
                states.type(torch.float32).contiguous()
            )  # (batch * block_size, n_embd)

        elif self.transformer_config.state_embedding_type.lower() == "grid":
            states = rearrange(
                states,
                "batch block height width channel -> (batch block) (channel height width)",
            )
            state_embeddings = self.state_embedding(
                states.type(torch.float32).contiguous()
            )  # (batch * block_size, n_embd)
        else:
            states = rearrange(
                states, "batch block state_dim -> (batch block) state_dim"
            )
            state_embeddings = self.state_embedding(
                states.type(torch.float32).contiguous()
            )
        state_embeddings = rearrange(
            state_embeddings,
            "(batch block) n_embd -> batch block n_embd",
            block=block_size,
        )
        return state_embeddings

    def get_action_embedding(self, actions):
        block_size = actions.shape[1]
        if block_size == 0:
            return None  # no actions to embed
        actions = rearrange(
            actions, "batch block action -> (batch block) action"
        )
        # I don't see why we need this but we do? Maybe because of the sequential?
        action_embeddings = self.action_embedding(actions).flatten(1)
        action_embeddings = rearrange(
            action_embeddings,
            "(batch block) n_embd -> batch block n_embd",
            block=block_size,
        )
        return action_embeddings

    def predict_states(self, x):
        return self.state_predictor(x)

    def predict_actions(self, x):
        return self.action_predictor(x)

    @abstractmethod
    def get_token_embeddings(
            self, state_embeddings, time_embeddings, action_embeddings, **kwargs
    ):
        """
        Returns the token embeddings for the transformer input.
        Note that different subclasses will have different token embeddings
        such as the DecisionTransformer which will use RTG (placed before the
        state embedding).

        Args:
            states: (batch, position, state_dim)
            actions: (batch, position)
            timesteps: (batch, position)
        Kwargs:
            rtgs: (batch, position) (only for DecisionTransformer)

        Returns:
            token_embeddings: (batch, position, n_embd)
        """
        pass

    @abstractmethod
    def get_action(self, **kwargs) -> int:
        """
        Returns the action given the state.
        """
        pass

    def initialize_time_embedding(self):
        if not (self.transformer_config.time_embedding_type == "linear"):
            self.time_embedding = nn.Embedding(
                self.environment_config.max_steps + 1,
                self.transformer_config.d_model,
            )
        else:
            self.time_embedding = nn.Linear(1, self.transformer_config.d_model)

        return self.time_embedding

    def initialize_state_embedding(self):
        # if self.transformer_config.state_embedding_type.lower() == "cnn":
        #     state_embedding = MiniGridConvEmbedder(
        #         self.transformer_config.d_model, endpool=True
        #     )
        # elif self.transformer_config.state_embedding_type.lower() == "vit":
        #     state_embedding = MiniGridViTEmbedder(
        #         self.transformer_config.d_model,
        #     )
        # else:
        if isinstance(self.environment_config.observation_space, Dict):
            n_obs = np.prod(
                self.environment_config.observation_space["image"].shape
            )
        else:
            n_obs = np.prod(
                self.environment_config.observation_space.shape
            )
        state_embedding = nn.Linear(
            n_obs, self.transformer_config.d_model, bias=False
        )

        nn.init.normal_(state_embedding.weight, mean=0.0, std=0.02)

        return state_embedding

    def initialize_state_predictor(self):
        if isinstance(self.environment_config.observation_space, Box):
            self.state_predictor = nn.Linear(
                self.transformer_config.d_model,
                np.prod(self.environment_config.observation_space.shape),
            )
        elif isinstance(self.environment_config.observation_space, Dict):
            self.state_predictor = nn.Linear(
                self.transformer_config.d_model,
                np.prod(
                    self.environment_config.observation_space["image"].shape
                ),
            )

    def initialize_easy_transformer(self):
        # Transformer
        cfg = HookedTransformerConfig(
            n_layers=self.transformer_config.n_layers,
            d_model=self.transformer_config.d_model,
            d_head=self.transformer_config.d_head,
            n_heads=self.transformer_config.n_heads,
            d_mlp=self.transformer_config.d_mlp,
            d_vocab=self.transformer_config.d_model,
            # 3x the max timestep so we have room for an action, reward, and state per timestep
            n_ctx=self.transformer_config.n_ctx + 6,
            act_fn=self.transformer_config.activation_fn,
            gated_mlp=self.transformer_config.gated_mlp,
            normalization_type=self.transformer_config.layer_norm,
            attention_dir="causal",
            d_vocab_out=self.transformer_config.d_model,
            seed=self.transformer_config.seed,
            device=self.transformer_config.device,
        )

        assert (
                cfg.attention_dir == "causal"
        ), "Attention direction must be causal"
        # assert cfg.normalization_type is None, "Normalization type must be None"

        transformer = HookedTransformer(cfg)

        # Because we passing in tokens, turn off embedding and update the position embedding
        transformer.embed = nn.Identity()
        transformer.pos_embed = PosEmbedTokens(cfg)
        # initialize position embedding
        nn.init.normal_(transformer.pos_embed.W_pos, cfg.initializer_range)
        # don't unembed, we'll do that ourselves.
        transformer.unembed = nn.Identity()

        return transformer


class DecisionTransformer(TrajectoryTransformer):
    def __init__(self, environment_config, transformer_config, device, **kwargs):
        super().__init__(
            environment_config=environment_config,
            transformer_config=transformer_config,
            **kwargs,
        )
        self.device = device
        self.model_type = "decision_transformer"
        self.reward_embedding = nn.Sequential(
            nn.Linear(1, self.transformer_config.d_model, bias=False)
        )
        self.mode_embedding = nn.Sequential(
            nn.Linear(2, self.transformer_config.d_model, bias=False)
        )
        self.reward_predictor = nn.Linear(self.transformer_config.d_model, 1)

        # n_ctx include full timesteps except for the last where it doesn't know the action
        assert (transformer_config.n_ctx - 2) % 3 == 0

        self.initialize_weights()

    def predict_rewards(self, x):
        return self.reward_predictor(x)

    def get_token_embeddings(
            self,
            state_embeddings,
            time_embeddings,
            reward_embeddings,
            action_embeddings=None,
            targets=None,
    ):
        """
        We need to compose the embeddings for:
            - states
            - actions
            - rewards
            - time

        Handling the cases where:
        1. we are training:
            1. we may not have action yet (reward, state)
            2. we have (action, state, reward)...
        2. we are evaluating:
            1. we have a target "a reward" followed by state

        1.1 and 2.1 are the same, but we need to handle the target as the initial reward.

        """
        batches = state_embeddings.shape[0]
        timesteps = time_embeddings.shape[1]

        reward_embeddings = reward_embeddings + time_embeddings
        state_embeddings = state_embeddings + time_embeddings

        if action_embeddings is not None:
            if action_embeddings.shape[1] < timesteps:
                assert (
                        action_embeddings.shape[1] == timesteps - 1
                ), "Action embeddings must be one timestep less than state embeddings"
                action_embeddings = (
                        action_embeddings
                        + time_embeddings[:, : action_embeddings.shape[1]]
                )
                trajectory_length = timesteps * 3 - 1
            else:
                action_embeddings = action_embeddings + time_embeddings
                trajectory_length = timesteps * 3
        else:
            trajectory_length = 2  # one timestep, no action yet

        if targets:
            targets = targets + time_embeddings

        # create the token embeddings
        token_embeddings = torch.zeros(
            (batches, trajectory_length, self.transformer_config.d_model),
            dtype=torch.float32,
            device=state_embeddings.device,
        )  # batches, blocksize, n_embd

        if action_embeddings is not None:
            token_embeddings[:, ::3, :] = reward_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings
        else:
            token_embeddings[:, 0, :] = reward_embeddings[:, 0, :]
            token_embeddings[:, 1, :] = state_embeddings[:, 0, :]

        if targets is not None:
            target_embedding = self.reward_embedding(targets)
            token_embeddings[:, 0, :] = target_embedding[:, 0, :]

        return token_embeddings

    def to_tokens(self, states, actions, rtgs, timesteps, mode=None):
        # embed states and recast back to (batch, block_size, n_embd)
        state_embeddings = self.get_state_embedding(
            states
        )  # batch_size, block_size, n_embd
        action_embeddings = (
            self.get_action_embedding(actions) if actions is not None else None
        )  # batch_size, block_size, n_embd or None
        reward_embeddings = self.get_reward_embedding(
            rtgs
        )  # batch_size, block_size, n_embd
        time_embeddings = self.get_time_embedding(
            timesteps
        )  # batch_size, block_size, n_embd

        # use state_embeddings, actions, rewards to go and
        token_embeddings = self.get_token_embeddings(
            state_embeddings=state_embeddings,
            action_embeddings=action_embeddings,
            reward_embeddings=reward_embeddings,
            time_embeddings=time_embeddings,
        )
        if mode is not None:
            # batch_size, seq_length = states.shape[0], states.shape[1]
            mode_seq_length = mode.shape[1]
            mode_embeddings = self.mode_embedding(mode.type(torch.float32))
            mode_stacked_inputs = torch.stack((mode_embeddings, mode_embeddings, mode_embeddings), dim=1)
            mode_stacked_inputs = mode_stacked_inputs.permute(0, 2, 1, 3)
            mode_stacked_inputs = mode_stacked_inputs.reshape(mode.shape[0], 3 * mode_seq_length,
                                                              self.transformer_config.d_model)
            # stacking the token_embeddings add mode
            token_embeddings = torch.cat((mode_stacked_inputs, token_embeddings), dim=1)

        return token_embeddings

    def get_action(self, states, actions, rewards, timesteps):
        state_preds, action_preds, reward_preds = self.forward(
            states, actions, rewards, timesteps
        )

        # get the action prediction
        action_preds = action_preds[:, -1, :]  # (batch, n_actions)
        action = torch.argmax(action_preds, dim=-1)  # (batch)
        return action

    def get_reward_embedding(self, rtgs):
        block_size = rtgs.shape[1]
        rtgs = rearrange(rtgs, "batch block rtg -> (batch block) rtg")
        rtg_embeddings = self.reward_embedding(rtgs.type(torch.float32))
        rtg_embeddings = rearrange(
            rtg_embeddings,
            "(batch block) n_embd -> batch block n_embd",
            block=block_size,
        )
        return rtg_embeddings

    def get_logits(self, x, batch_size, seq_length, no_actions: bool, mode=None):
        if no_actions is False:
            # TODO replace with einsum
            if (x.shape[1] % 3 != 0) and ((x.shape[1] + 1) % 3 == 0):
                x = torch.concat((x, x[:, -2].unsqueeze(1)), dim=1)

            if mode is None:
                x = x.reshape(
                    batch_size, seq_length, 3, self.transformer_config.d_model
                )
            else:
                x = x.reshape(
                    batch_size, seq_length, -1, 3, self.transformer_config.d_model
                )

            x = x.permute(0, 3, 1, 2, 4)
            x = rearrange(x, "batch b seq_len a d -> batch b seq_len (a d)")

            # predict next return given state and action
            reward_preds = None #self.predict_rewards(x[:, 2])
            # predict next state given state and action
            state_preds = None #self.predict_states(x[:, 2])
            # predict next action given state and RTG
            action_preds = self.predict_actions(x[:, 1])

            return state_preds, action_preds, reward_preds

        else:
            # TODO replace with einsum
            x = x.reshape(
                batch_size, seq_length, 2, self.transformer_config.d_model
            )
            x = x.permute(0, 2, 1, 3)
            # predict next action given state and RTG
            action_preds = self.predict_actions(x[:, 1])
            return None, action_preds, None

    def forward(
            self,
            # has variable shape, starting with batch, position
            states: TT[...],  # noqa: F821
            actions: TT["batch", "position"],  # noqa: F821
            rtgs: TT["batch", "position"],  # noqa: F821
            timesteps: TT["batch", "position"],  # noqa: F821
            mode: TT["batch", "position"],  # noqa: F821
            # pad_action: bool = True,

    ) -> Tuple[
        TT[...], TT["batch", "position"], TT["batch", "position"]  # noqa: F821
    ]:
        batch_size = states.shape[0]
        seq_length = states.shape[1]
        no_actions = actions is None

        if no_actions is False:
            if actions.shape[1] < seq_length - 1:
                raise ValueError(
                    f"Actions required for all timesteps except the last, got {actions.shape[1]} and {seq_length}"
                )

            # if actions.shape[1] == seq_length - 1:
            #     if pad_action:
            #         print(
            #             "Warning: actions are missing for the last timestep, padding with zeros")
            #         # This means that you can't interpret Reward or State predictions for the last timestep!!!
            #         actions = torch.cat([actions, torch.zeros(
            #             batch_size, 1, 1, dtype=torch.long, device=actions.device)], dim=1)

        states = states.to(device=self.device)
        actions = actions.to(device=self.device)
        rtgs = rtgs.to(device=self.device)
        timesteps = timesteps.to(device=self.device)
        mode = mode.to(device=self.device) if mode is not None else None
        # embed states and recast back to (batch, block_size, n_embd)
        token_embeddings = self.to_tokens(states, actions, rtgs, timesteps, mode)
        x = self.transformer(token_embeddings)
        # print("output", x.shape, seq_length)
        state_preds, action_preds, reward_preds = self.get_logits(
            x, batch_size, seq_length, no_actions=no_actions, mode=mode
        )
        return state_preds, action_preds, reward_preds


class PosEmbedTokens(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.cfg.n_ctx = self.cfg.n_ctx
        self.W_pos = nn.Parameter(
            torch.empty(self.cfg.n_ctx + 6, self.cfg.d_model)
        )

    def forward(
            self,
            tokens: TT["batch", "position"],  # noqa: F821
            past_kv_pos_offset: int = 0,
    ) -> TT["batch", "position", "d_model"]:  # noqa: F821
        """Tokens have shape [batch, pos]
        Output shape [pos, d_model] - will be broadcast along batch dim"""

        tokens_length = tokens.size(-2)
        pos_embed = self.W_pos[:tokens_length, :]  # [pos, d_model]
        broadcast_pos_embed = einops.repeat(
            pos_embed, "pos d_model -> batch pos d_model", batch=tokens.size(0)
        )  # [batch, pos, d_model]
        return broadcast_pos_embed
