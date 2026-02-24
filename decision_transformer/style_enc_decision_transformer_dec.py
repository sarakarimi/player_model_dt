from abc import abstractmethod
from typing import Tuple, Union, Dict, Optional
import einops
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from gymnasium.spaces import Box, Dict
from torchtyping import TensorType as TT
from transformer_lens import HookedTransformer, HookedTransformerConfig
from configs import EnvironmentConfig, TransformerModelConfig, TransformerEncoderConfig



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
            self.transformer_config.d_model, environment_config.action_space.n
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
            # states = rearrange(
            #     states,
            #     "batch block grid -> (batch block) grid",
            # )
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
            n_ctx=self.transformer_config.n_ctx + 1 + 3,  # 1 accounts for the style token / prompt token
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


class StyleDecisionTransformer(TrajectoryTransformer):
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
        self.style_embedding = nn.Linear(environment_config.style_vector_size, self.transformer_config.d_model,
                                         bias=False)

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

    def to_tokens(self, states, actions, rtgs, timesteps, prompt=None):
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
        if prompt is not None:
            batch_size, seq_length = states.shape[0], states.shape[1]
            style_embeddings = prompt #self.style_embedding(prompt.type(torch.float32))
            style_stacked_inputs = style_embeddings.reshape(batch_size, 3, self.transformer_config.d_model)

            assert style_stacked_inputs.shape == (batch_size, 3, self.transformer_config.d_model)

            # stacking the token_embeddings add mode
            token_embeddings = torch.cat((style_stacked_inputs, token_embeddings), dim=1)
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

    def get_logits(self, x, batch_size, seq_length, no_actions: bool, prompt=None):
        if no_actions is False:
            # TODO replace with einsum
            if (x.shape[1] % 3 != 0) and ((x.shape[1] + 1) % 3 == 0):
                x = torch.concat((x, x[:, -2].unsqueeze(1)), dim=1)
            if prompt is None:
                x = x.reshape(
                    batch_size, seq_length, 3, self.transformer_config.d_model
                )
            else:
                x = x[:, 3:, :]  # remove style_vector or prompt token
                x = x.reshape(
                    batch_size, -1, 3, self.transformer_config.d_model
                )

            x = x.permute(0, 2, 1, 3)  # x.permute(0, 3, 1, 2, 4)
            # x = rearrange(x, "batch b seq_len a d -> batch b seq_len (a d)")

            # predict next return given state and action
            reward_preds = None  # self.predict_rewards(x[:, 2])
            # predict next state given state and action
            state_preds = None  # self.predict_states(x[:, 2])
            # predict next action given state and RTG
            action_preds =   self.predict_actions(x[:, 1, :, :])  #self.predict_actions(x[:, 1])[:, -seq_length:, :] #
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
            prompt: TT["batch", "position"],  # noqa: F821
            pad_action: bool = True,

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

            if actions.shape[1] == seq_length - 1:
                if pad_action:
                    # print(
                    #     "Warning: actions are missing for the last timestep, padding with zeros")
                    # This means that you can't interpret Reward or State predictions for the last timestep!!!
                    actions = torch.cat([actions, torch.zeros(
                        batch_size, 1, 1, dtype=torch.long, device=actions.device)], dim=1)

        states = states.to(device=self.device)
        actions = actions.to(device=self.device)
        rtgs = rtgs.to(device=self.device)
        timesteps = timesteps.to(device=self.device)
        prompt = prompt.to(device=self.device) if prompt is not None else None
        # embed states and recast back to (batch, block_size, n_embd)

        # batch_mode_percentage = [1 for m in mode if m[0][0] == 0.]
        # batch_mode_percentage = sum(batch_mode_percentage) / len(mode)
        # print("BATCH MODE PERCENTAGE", batch_mode_percentage)

        token_embeddings = self.to_tokens(states, actions, rtgs, timesteps, prompt)
        x = self.transformer(token_embeddings)
        state_preds, action_preds, reward_preds = self.get_logits(
            x, batch_size, seq_length, no_actions=no_actions, prompt=prompt
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
            torch.empty(self.cfg.n_ctx, self.cfg.d_model)
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


class TransformerEncoder(nn.Module):
    """
    Variable-length (padded) Transformer Encoder.

    Inputs:
      x:    (B, S_max, 9)
      mask: (B, S_max) True=valid, False=PAD
    Outputs:
      x_hat: (B, S_max, 9) (you'll usually ignore PAD positions via mask)
    """

    def __init__(self, cfg: TransformerEncoderConfig, dt: TrajectoryTransformer):
        super().__init__()

        self.cfg = cfg
        if cfg.dim_ff is None:
            cfg.dim_ff = 4 * cfg.d_model

        self.dt = dt

        # Trajectory encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.style_encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_enc_layers)
        self.enc_ln = nn.LayerNorm(cfg.d_model)

        # VAE layers
        self.to_mu = nn.Linear(cfg.d_model, cfg.z_dim)
        self.to_logvar = nn.Linear(cfg.d_model, cfg.z_dim)


    def encode(
        self,
        full_states: torch.Tensor,      # [B, S, ...] (can be flat or image)
        full_actions: torch.Tensor,     # [B, S, act_dim]
        full_timesteps: torch.Tensor,   # [B, S]
        full_attn_mask: torch.Tensor,   # [B, S]
    ):
        """Encode full trajectory into latent style z."""
        batch_size = full_states.shape[0]
        seq_len = full_states.shape[1]

        # Embed states and actions
        # Note: get_state_embedding handles both flat and image states internally
        s_emb = self.dt.get_state_embedding(full_states)

        # Embed discrete actions directly
        if full_actions.ndim == 3:
            full_actions = full_actions.squeeze(-1)
        full_actions = torch.clamp(full_actions.long(), 0, self.dt.environment_config.action_space.n - 1)
        # get_action_embedding expects 3D input [batch, block, action]
        a_emb = self.dt.get_action_embedding(full_actions.unsqueeze(-1))

        # get_time_embedding expects 3D input [batch, block, time]
        t_emb = self.dt.get_time_embedding(full_timesteps.unsqueeze(-1))

        # Add time embeddings
        s_emb = s_emb + t_emb
        a_emb = a_emb + t_emb

        # Stack as [s_t, a_t] for each timestep
        tokens = torch.stack((s_emb, a_emb), dim=1)  # [B, 2, S, H]
        tokens = tokens.permute(0, 2, 1, 3).reshape(batch_size, 2 * seq_len, self.cfg.d_model)

        # Token-level attention mask
        token_mask = torch.stack((full_attn_mask, full_attn_mask), dim=1)
        token_mask = token_mask.permute(0, 2, 1).reshape(batch_size, 2 * seq_len)

        # Run encoder
        src_key_padding_mask = (token_mask == 0)  # True = padding
        h = self.style_encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        h = self.enc_ln(h)

        # Masked mean pooling
        m = token_mask.unsqueeze(-1).to(h.dtype)
        denom = m.sum(dim=1).clamp_min(1.0)
        pooled = (h * m).sum(dim=1) / denom

        # VAE parameters
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)

        return mu, logvar



class VariationalStyleDecisionTransformer(nn.Module):
    """
    VAE-style global z per episode, DT as autoregressive decoder.

    - Encoder: TransformerEncoder -> (mu, logvar)
    - Reparam: z = mu + eps * sigma
    - Decoder: your StyleDecisionTransformer, using `prompt=z` to inject style token
    - Loss: CE(action) + beta * KL
    """

    def __init__(
            self,
            transformer_config: TransformerModelConfig,
            environment_config: EnvironmentConfig,
            encoder_config: TransformerEncoderConfig,
            device: torch.device,
            beta: float = 0.0085,
           ):
        super().__init__()
        self.transformer_config = transformer_config
        self.environment_config = environment_config
        self.encoder_config = encoder_config
        self.device = device
        self.beta = beta



        self.dt = StyleDecisionTransformer(
            environment_config=environment_config,
            transformer_config=transformer_config,
            device=device,
        )

        self.encoder = TransformerEncoder(cfg=encoder_config, dt=self.dt)

        # Latent to style tokens
        self.z_to_style_tokens = nn.Sequential(
            nn.Linear(encoder_config.z_dim, 3 * encoder_config.d_model),
            nn.GELU(),
            nn.Linear(3 * encoder_config.d_model, 3 * encoder_config.d_model),
            nn.LayerNorm(3 * encoder_config.d_model),
        )


        assert (
                self.dt.style_embedding.in_features == self.encoder.cfg.z_dim
        ), f"DT style_vector_size ({self.dt.style_embedding.in_features}) must match encoder z_dim ({self.encoder.cfg.z_dim})"

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # mu/logvar: (B, z_dim)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl_loss = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1.0 - logvar, dim=-1)
        kl_loss = kl_loss.mean()
        weighted_kl_loss = self.beta * kl_loss

        return weighted_kl_loss

    def forward(
            self,
            # DT inputs:
            states: torch.Tensor,
            actions: Optional[torch.Tensor],
            rtgs: torch.Tensor,
            timesteps: torch.Tensor,
            # Encoder inputs:
            full_states: torch.Tensor,
            full_actions: torch.Tensor,
            full_timesteps: torch.Tensor,
            full_attn_masks: torch.Tensor,
            sample_latent: bool = True,
    ) -> dict[str, torch.Tensor]:
        # Encoder
        mu, logvar = self.encoder.encode(full_states, full_actions, full_timesteps, full_attn_masks)  # (B,z), (B,z)

        # Sample z and convert latent z to style tokens [B, 3, H]
        if self.training and sample_latent:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        batch_size = z.size(0)
        x = self.z_to_style_tokens(z)
        prompt = x.view(batch_size, 3, self.encoder_config.d_model)

        # DT decoder
        _, action_preds, _ = self.dt(
            states=states,
            actions=actions,
            rtgs=rtgs,
            timesteps=timesteps,
            prompt=prompt,
        )

        return {
            "action_preds": action_preds,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }




