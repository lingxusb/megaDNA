import torch
import torch.nn.functional as F
from itertools import zip_longest
from tqdm.auto import tqdm
from einops import rearrange, reduce, repeat
from beartype import beartype
from beartype.typing import Tuple, Union

from MEGABYTE_pytorch import MEGABYTE
from MEGABYTE_pytorch.megabyte import reduce_mult, remainder_to_mult, default, exists
from MEGABYTE_pytorch.megabyte import pack_one, unpack_one, top_k, gumbel_sample

class MEGADNA(MEGABYTE):

    @beartype
    def __init__(
        self,
        *,
        num_tokens,
        dim: Union[Tuple, int],
        depth: Tuple,
        max_seq_len: Tuple,
    ):
        super().__init__(
            num_tokens = num_tokens,
            dim = dim,
            depth = depth,
            max_seq_len = max_seq_len,
            dim_head = 64,
            heads = 8,
            attn_dropout = 0.,
            ff_mult = 4,
            ff_dropout = 0.,
            pad_id = 0,
            rel_pos = False,
            pos_emb = False,
            flash_attn = True
        )

    def generate(self, prime = None, seq_len = 1024, filter_thres = 0.9, temperature = 1., default_batch_size = 1):
        device = next(self.parameters()).device

        seq = prime if exists(prime) else torch.empty((default_batch_size, 0), dtype = torch.long, device = device)
        batch = seq.shape[0]
        
        #### half precision to save memory
#         self.half()
        with torch.no_grad():
            for _ in tqdm(range(seq_len - seq.shape[-1])):
                logits = self.forward(seq, return_value='logits')[:, -1]
                logits = top_k(logits, thres = filter_thres)
                sampled = gumbel_sample(logits, dim = -1, temperature = temperature)
                seq = torch.cat((seq, rearrange(sampled, 'b -> b 1')), dim = -1)
                del logits, sampled

        return seq.reshape(batch, -1)

    def forward(self, ids, return_value = 'loss'):

        if return_value not in ['logits', 'embedding', 'loss']:
            raise ValueError('return_value must be one of "embedding", "logits", or "loss"')
        
        batch = ids.shape[0]

        assert ids.ndim in {2, self.stages + 1}
        flattened_dims = ids.ndim == 2
        ids_orig_ndim = ids.ndim

        if ids.numel() == 0:
            return self.forward_empty(ids.shape[0])

        if flattened_dims:
            # allow for ids to be given in the shape of (batch, seq)
            # in which case it will be auto-padded to the next nearest multiple of depth seq len
            seq_len = ids.shape[-1]
            multiple_of = reduce_mult(self.max_seq_len[1:])
            padding = remainder_to_mult(seq_len, multiple_of)
            ids = F.pad(ids, (0, padding), value = self.pad_id)
            ids = ids.reshape(batch, -1, *self.max_seq_len[1:])

        b, *prec_dims, device = *ids.shape, ids.device

        # check some dimensions

        assert prec_dims[0] <= self.max_seq_len[0], 'the first dimension of your axial autoregressive transformer must be less than the first tuple element of max_seq_len (like any autoregressive transformer)'
        assert tuple(prec_dims[1:]) == tuple(self.max_seq_len[1:]), 'all subsequent dimensions must match exactly'

        # get tokens for all hierarchical stages, reducing by appropriate dimensions
        # and adding the absolute positional embeddings

        tokens_at_stages = []
        pos_embs = default(self.pos_embs, (None,))

        for ind, pos_emb, token_emb in zip_longest(range(len(prec_dims)), pos_embs, self.token_embs):
            is_first = ind == 0

            tokens = token_emb(ids)
            # print(f"ids shape is {ids.shape}")
            # print(f"tokens shape is {tokens.shape}")

            if exists(pos_emb):
                positions = pos_emb(torch.arange(tokens.shape[-2], device = device))
                tokens = tokens + positions

            tokens_at_stages.insert(0, tokens)

            if is_first:
                continue

            ids = rearrange(ids, '... m n -> ... (m n)')

        # the un-pixelshuffled representations of the previous hierarchy, starts with None

        prev_stage_tokens_repr = None
        hidden_states = []

        # spatial tokens is tokens with depth pos reduced along depth dimension + spatial positions        

        for stage_start_tokens, stage_tokens, transformer, proj in zip(self.start_tokens, tokens_at_stages, self.transformers, self.to_next_transformer_projections):
            # print(f" starge start token shape is {stage_start_tokens.shape}")
            # print(f" stage token shape is {stage_tokens.shape}")
            stage_tokens, ps = pack_one(stage_tokens, '* n d')
            stage_start_tokens = repeat(stage_start_tokens, 'f -> b 1 f', b = stage_tokens.shape[0])

            # concat start token

            stage_tokens = torch.cat((
                stage_start_tokens,
                stage_tokens,
            ), dim = -2)

            # sum the previous hierarchy's representation
            if exists(prev_stage_tokens_repr):
                prev_stage_tokens_repr = F.pad(prev_stage_tokens_repr, (0, 0, 1, 0), value = 0.)
                stage_tokens = stage_tokens + prev_stage_tokens_repr

            attended = transformer(stage_tokens)
            hidden_states.append(attended)

            attended = unpack_one(attended, ps, '* n d')

            # project for next stage in the hierarchy

            prev_stage_tokens_repr = proj(attended[..., :-1, :])

        if return_value == 'embedding':
            return hidden_states
            
        # project to logits

        logits = self.to_logits(attended)

        start_tokens = logits[(slice(None), *((0,) * (logits.ndim - 2)), slice(None))]
        start_tokens = rearrange(start_tokens, 'b d -> b 1 d')

        logits = logits[..., 1:, :]

        if return_value == 'logits':

            if flattened_dims:
                logits = rearrange(logits, 'b ... c -> b (...) c')
                logits = logits[:, :seq_len]

            return logits
            
        logits = rearrange(logits, 'b ... c -> b (...) c')
        logits = torch.cat((start_tokens, logits), dim = -2)

        preds = rearrange(logits, 'b n c -> b c n')
        labels = rearrange(ids, 'b ... -> b (...)')

        loss = F.cross_entropy(
            preds[..., :-1],
            labels,
            ignore_index = self.pad_id
        )

        return loss