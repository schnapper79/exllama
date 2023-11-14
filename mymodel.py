import gc

import torch
import torch.nn.functional as F
from torch import version as torch_version

from generator import ExLlamaGenerator
from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
import os, glob


def get_max_prompt_length(state, tl):
    return tl - state['max_new_tokens']

def clear_torch_cache():
    gc.collect()
    torch.cuda.empty_cache()

class ExllamaModel:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, path_to_model,cfg):
        tokenizer_path = os.path.join(path_to_model, "tokenizer.model")
        model_config_path = os.path.join(path_to_model, "config.json")
        st_pattern = os.path.join(path_to_model, "*.safetensors")
        model_path = glob.glob(st_pattern)

        config = ExLlamaConfig(str(model_config_path))
        config.model_path = str(model_path)
        config.max_seq_len = cfg['max_seq_len']
        config.compress_pos_emb = cfg['compress_pos_emb']
        if cfg['gpu_split']:
            config.set_auto_map(cfg['gpu_split'])
            config.gpu_peer_fix = True

        if cfg['alpha_value'] > 1 and cfg['rope_freq_base'] == 0:
            config.alpha_value = cfg['alpha_value']
            config.calculate_rotary_embedding_base()
        elif cfg['rope_freq_base'] > 0:
            config.rotary_embedding_base = cfg['rope_freq_base']

        if torch_version.hip:
            config.rmsnorm_no_half2 = True
            config.rope_no_half2 = True
            config.matmul_no_half2 = True
            config.silu_no_half2 = True

        model = ExLlama(config)
        tokenizer = ExLlamaTokenizer(str(tokenizer_path))
        cache = ExLlamaCache(model)
        generator = ExLlamaGenerator(model, tokenizer, cache)

        result = cls()
        result.config = config
        result.model = model
        result.cache = cache
        result.tokenizer = tokenizer
        result.generator = generator
        return result

    def encode(self, string, **kwargs):
        return self.tokenizer.encode(string, max_seq_len=self.model.config.max_seq_len, add_bos=True)

    def decode(self, ids, **kwargs):
        if isinstance(ids, list):
            ids = torch.tensor([ids])
        elif isinstance(ids, torch.Tensor) and ids.numel() == 1:
            ids = ids.view(1, -1)

        return self.tokenizer.decode(ids)[0]

    def get_logits(self, token_ids, **kwargs):
        self.cache.current_seq_len = 0
        if token_ids.shape[-1] > 1:
            self.model.forward(token_ids[:, :-1], self.cache, input_mask=None, preprocess_only=True)

        return self.model.forward(token_ids[:, -1:], self.cache, **kwargs).float().cpu()

    def generate_with_streaming(self, prompt, state):

        # The cache batch size must be 2 for CFG and 1 otherwise
        if state['guidance_scale'] == 1:
            if self.cache.batch_size == 2:
                del self.cache
                clear_torch_cache()
                self.cache = ExLlamaCache(self.model)
                self.generator = ExLlamaGenerator(self.model, self.tokenizer, self.cache)
        else:
            if self.cache.batch_size == 1:
                del self.cache
                clear_torch_cache()
                self.cache = ExLlamaCache(self.model, batch_size=2)
                self.generator = ExLlamaGenerator(self.model, self.tokenizer, self.cache)

        self.generator.settings.temperature = state['temperature']
        self.generator.settings.top_p = state['top_p']
        self.generator.settings.top_k = state['top_k']
        self.generator.settings.typical = state['typical_p']
        self.generator.settings.token_repetition_penalty_max = state['repetition_penalty']
        self.generator.settings.token_repetition_penalty_sustain = -1 if state['repetition_penalty_range'] <= 0 else state['repetition_penalty_range']
        if state['ban_eos_token']:
            self.generator.disallow_tokens([self.tokenizer.eos_token_id])
        else:
            self.generator.disallow_tokens(None)

        if state['custom_token_bans']:
            to_ban = [int(x) for x in state['custom_token_bans'].split(',')]
            if len(to_ban) > 0:
                self.generator.disallow_tokens(to_ban)

        # Case 1: no CFG
        if state['guidance_scale'] == 1:
            self.generator.end_beam_search()

            # Tokenizing the input
            ids = self.generator.tokenizer.encode(prompt, max_seq_len=self.model.config.max_seq_len)
            if state['add_bos_token']:
                ids = torch.cat(
                    [torch.tensor([[self.tokenizer.bos_token_id]]).to(ids.device),
                     ids], dim=1
                ).to(torch.int64)
            ids = ids[:, -get_max_prompt_length(state,self.config.max_seq_len-4):]
            if state['auto_max_new_tokens']:
                max_new_tokens = state['truncation_length'] - ids.shape[-1]
            else:
                max_new_tokens = state['max_new_tokens']

            self.generator.gen_begin_reuse(ids)
            initial_len = self.generator.sequence[0].shape[0]
            has_leading_space = False

            for i in range(max_new_tokens):
                token = self.generator.gen_single_token()
                if i == 0 and self.generator.tokenizer.tokenizer.IdToPiece(int(token)).startswith('▁'):
                    has_leading_space = True

                decoded_text = self.generator.tokenizer.decode(self.generator.sequence[0][initial_len:])
                if has_leading_space:
                    decoded_text = ' ' + decoded_text

                yield decoded_text
                if token.item() == self.generator.tokenizer.eos_token_id or shared.stop_everything:
                    break

        # Case 2: CFG
        # Copied from https://github.com/turboderp/exllama/blob/master/example_cfg.py
        else:
            alpha = state['guidance_scale']
            prompts = [prompt, state['negative_prompt'] or '']

            ids, mask = self.tokenizer.encode(
                prompts,
                return_mask=True,
                max_seq_len=self.model.config.max_seq_len,
                add_bos=state['add_bos_token']
            )
            if state['auto_max_new_tokens']:
                max_new_tokens = state['truncation_length'] - ids[0].shape[-1]
            else:
                max_new_tokens = state['max_new_tokens']

            self.generator.gen_begin(ids, mask=mask)
            initial_len = self.generator.sequence[0].shape[0]
            has_leading_space = False

            for i in range(max_new_tokens):
                logits = self.model.forward(self.generator.sequence[:, -1:], self.cache, input_mask=mask)
                self.generator.apply_rep_penalty(logits)

                logits = F.log_softmax(logits, dim=-1)
                logits_mixed = alpha * logits[0] + (1 - alpha) * logits[1]

                token, _ = self.generator.sample_current(logits_mixed)
                if i == 0 and self.generator.tokenizer.tokenizer.IdToPiece(int(token)).startswith('▁'):
                    has_leading_space = True

                decoded_text = self.generator.tokenizer.decode(self.generator.sequence[0][initial_len:])
                if has_leading_space:
                    decoded_text = ' ' + decoded_text

                yield decoded_text
                if token.item() == self.tokenizer.eos_token_id or shared.stop_everything:
                    break

                batch_token = token.repeat(2, 1)
                self.generator.gen_accept_token(batch_token)

    def generate(self, prompt, state):
        output = ''
        for output in self.generate_with_streaming(prompt, state):
            pass

        return output