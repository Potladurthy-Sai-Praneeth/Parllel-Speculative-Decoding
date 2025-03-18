# import torch
# import math
# from .util import norm_logits, sample


# class KVCacheModel():
#     def __init__(self, model1 : torch.nn.Module, model2 : torch.nn.Module, temperature : float = 1, top_k : int = 0, top_p : float = 0) -> None:
#         self._model1 = model1
#         self._model2 = model2
#         self._past_key_values = None
#         self._prob_history = None

#         self._temperature = temperature
#         self._top_k = top_k
#         self._top_p = top_p

#     def _forward_with_kvcache(self, input_ids : torch.Tensor) -> torch.Tensor:
#         if self._past_key_values is None:
#             outputs = self._model(input_ids)
#             self._prob_history = outputs.logits[:, :, :self.vocab_size]
#             for i in range(self._prob_history.shape[-2]):   
#                 self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
#             self._past_key_values = outputs.past_key_values
#             last_q = self._prob_history[:, -1, :]
#         else:
#             # return the last token's logits
#             cached_len = self._past_key_values[0][0].shape[2]
                
#             last_input_id = input_ids[:, cached_len:]
#             if last_input_id.dim() == 1:
#                 last_input_id = torch.unsqueeze(last_input_id, 0)
            
#             outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            
#             not_cached_q = outputs.logits[:, :, :self.vocab_size]
            
#             if not_cached_q.dim() == 2:
#                 not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
#             for i in range(not_cached_q.shape[-2]):   
#                 not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
                
#             self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            
#             last_q = not_cached_q[:, -1, :]
#             self._past_key_values = outputs.past_key_values
        
#         return last_q


#     def _generate_with_kvcache(self, prefix : torch.Tensor, 
#                                     gamma : int) -> torch.Tensor:
#         """ forward the model gamma times

#         Args:
#             prefix (torch.Tensor): the prefix
#             gamma (int): how many times approx guesses

#         Returns:
#             Torch.Tensor: prefix+generated tokens
#         """
#         x = prefix.to(self._model2.device)
#         self._model = self._model2
#         if self._prob_history is not None:
#             self._prob_history = self._prob_history.to(self._model2.device)
#         if self._past_key_values is not None:
#             self._past_key_values = [(k.to(self._model2.device), v.to(self._model2.device)) for k, v in self._past_key_values]
        
#         model_1_generate_num = math.ceil(gamma / 2)

#         for _ in range(model_1_generate_num):
#             q = self._forward_with_kvcache(x)
#             next_tok = sample(q)
#             x = torch.cat((x, next_tok), dim=1)
        
#         x = x.to(self._model1.device)
#         self._model = self._model1
#         self._prob_history = self._prob_history.to(self._model1.device)
#         self._past_key_values = [(k.to(self._model1.device), v.to(self._model1.device)) for k, v in self._past_key_values]

#         for _ in range(gamma - model_1_generate_num):
#             q = self._forward_with_kvcache(x)
#             next_tok = sample(q)
#             x = torch.cat((x, next_tok), dim=1)
#         return x

#     @torch.no_grad()
#     def generate(self, input : torch.Tensor, gamma : int) -> torch.Tensor:
#         output = self._generate_with_kvcache(input, gamma)
#         return output
    
#     @torch.no_grad()
#     def rollback(self, end_pos : int):
#         past_key_values_trimmed = []
#         assert self._past_key_values
#         for kv in self._past_key_values:
#             k, v = kv
#             k = k[:, :, :end_pos, :]
#             v = v[:, :, :end_pos, :]
#             kv_trimmed = (k, v)
#             past_key_values_trimmed.append(kv_trimmed)
        
#         self._past_key_values = past_key_values_trimmed
#         self._prob_history = self._prob_history[:, :end_pos, :]


import torch
import math
from .util import norm_logits, sample
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


# class KVCacheModel():
#     def __init__(self, models: list, temperature: float = 1, top_k: int = 0, top_p: float = 0) -> None:
#         """
#         Initialize KVCacheModel with multiple models for token generation
        
#         Args:
#             models (list): List of models to use for generation
#             temperature (float): Temperature for sampling
#             top_k (int): Top-k sampling parameter
#             top_p (float): Top-p (nucleus) sampling parameter
#         """
#         self._models = models
#         self._current_model_idx = 0
#         self._model = self._models[self._current_model_idx]
#         self._past_key_values = None
#         self._prob_history = None

#         self._temperature = temperature
#         self._top_k = top_k
#         self._top_p = top_p
        


#     def _forward_with_kvcache(self, input_ids: torch.Tensor) -> torch.Tensor:
#         """Forward pass using KV cache for efficient inference"""
#         if self._past_key_values is None:
#             outputs = self._model(input_ids)
#             self._prob_history = outputs.logits[:, :, :self.vocab_size]
#             for i in range(self._prob_history.shape[-2]):   
#                 self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
#             self._past_key_values = outputs.past_key_values
#             last_q = self._prob_history[:, -1, :]
#         else:
#             # return the last token's logits
#             cached_len = self._past_key_values.get_seq_length()
                
#             last_input_id = input_ids[:, cached_len:]
#             if last_input_id.dim() == 1:
#                 last_input_id = torch.unsqueeze(last_input_id, 0)
            
#             outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            
#             not_cached_q = outputs.logits[:, :, :self.vocab_size]
            
#             if not_cached_q.dim() == 2:
#                 not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
#             for i in range(not_cached_q.shape[-2]):   
#                 not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
                
#             self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            
#             last_q = not_cached_q[:, -1, :]
#             self._past_key_values = outputs.past_key_values
        
#         return last_q

#     def _switch_to_model(self, model_idx):
#         """Switch active model and move KV cache to its device"""
#         if model_idx >= len(self._models):
#             model_idx = 0  # Wrap around to first model if index exceeds model count
            
#         self._current_model_idx = model_idx
#         self._model = self._models[model_idx]
        
#         # Move prob history and KV cache to the current model's device
#         if self._prob_history is not None:
#             self._prob_history = self._prob_history.to(self._model.device)
        
#         if self._past_key_values is not None:
#             self._past_key_values = [(k.to(self._model.device), v.to(self._model.device)) 
#                                     for k, v in self._past_key_values]

#     def _generate_with_kvcache(self, prefix: torch.Tensor, gamma: int) -> torch.Tensor:
#         """ 
#         Forward multiple models to generate tokens, cycling through available models
        
#         Args:
#             prefix (torch.Tensor): the prefix tokens
#             gamma (int): how many tokens to generate
            
#         Returns:
#             torch.Tensor: prefix + generated tokens
#         """
#         x = prefix.to(self._models[0].device)
        
#         # Reset to first model
#         self._switch_to_model(0)
        
#         # Calculate tokens per model (distribute as evenly as possible)
#         tokens_per_model = [gamma // len(self._models)] * len(self._models)
#         # Distribute remaining tokens
#         for i in range(gamma % len(self._models)):
#             tokens_per_model[i] += 1
            
#         for model_idx, tokens_to_generate in enumerate(tokens_per_model):
#             self._switch_to_model(model_idx)
            
#             for _ in range(tokens_to_generate):
#                 q = self._forward_with_kvcache(x)
#                 next_tok = sample(q)
#                 x = torch.cat((x, next_tok), dim=1)
                
#         return x

#     @torch.no_grad()
#     def generate(self, input: torch.Tensor, gamma: int) -> torch.Tensor:
#         """
#         Generate tokens using multiple models
        
#         Args:
#             input (torch.Tensor): Input token ids
#             gamma (int): Number of tokens to generate
            
#         Returns:
#             torch.Tensor: Original input + generated tokens
#         """
#         output = self._generate_with_kvcache(input, gamma)
#         return output
    
#     @torch.no_grad()
#     def rollback(self, end_pos: int):
#         """
#         Roll back KV cache to a previous position
        
#         Args:
#             end_pos (int): Position to trim the cache to
#         """
#         # past_key_values_trimmed = []
#         # assert self._past_key_values
#         # for kv in self._past_key_values:
#         #     k, v = kv
#         #     k = k[:, :, :end_pos, :]
#         #     v = v[:, :, :end_pos, :]
#         #     kv_trimmed = (k, v)
#         #     past_key_values_trimmed.append(kv_trimmed)
        
#         # self._past_key_values = past_key_values_trimmed
#         # self._prob_history = self._prob_history[:, :end_pos, :]
#         self._past_key_values.crop(end_pos)
#         self._prob_history = self._prob_history[:, :end_pos, :]

class KVCacheModel:
    def __init__(self, models: list, temperature: float = 1, top_k: int = 0, top_p: float = 0) -> None:
        self._models = models
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self.vocab_size = None  # Set externally
        self._prob_history = None  # Aggregated probability history

        # Per-model states (KV cache and prob history)
        self._model_states = [
            {'past_key_values': None, 'prob_history': None} 
            for _ in models
        ]

    def _switch_to_model(self, model_idx):
        """Switch active model without resetting state."""
        if model_idx >= len(self._models):
            model_idx = 0
        self._current_model_idx = model_idx

    # def _forward_with_kvcache(self, input_ids: torch.Tensor) -> torch.Tensor:   # Deep seek op (Issue with get_seq_length) 
    #     """Forward pass using the current model's KV cache."""
    #     model_idx = self._current_model_idx
    #     model = self._models[model_idx]
    #     state = self._model_states[model_idx]
        
    #     if state['past_key_values'] is None:
    #         outputs = model(input_ids)
    #         state['prob_history'] = outputs.logits[:, :, :self.vocab_size]
    #         for i in range(state['prob_history'].shape[-2]):
    #             state['prob_history'][:, i, :] = norm_logits(
    #                 state['prob_history'][:, i, :], 
    #                 self._temperature, 
    #                 self._top_k, 
    #                 self._top_p
    #             )
    #         state['past_key_values'] = outputs.past_key_values
    #         last_q = state['prob_history'][:, -1, :]
    #     else:
    #         cached_len = state['past_key_values'][0][0].shape[2]
    #         last_input_id = input_ids[:, cached_len:]
    #         outputs = model(
    #             last_input_id, 
    #             past_key_values=state['past_key_values'], 
    #             use_cache=True
    #         )
    #         not_cached_q = outputs.logits[:, :, :self.vocab_size]
    #         for i in range(not_cached_q.shape[-2]):
    #             not_cached_q[:, i, :] = norm_logits(
    #                 not_cached_q[:, i, :], 
    #                 self._temperature, 
    #                 self._top_k, 
    #                 self._top_p
    #             )
    #         state['prob_history'] = torch.cat(
    #             [state['prob_history'], not_cached_q], 
    #             dim=1
    #         ) if state['prob_history'] is not None else not_cached_q
    #         last_q = not_cached_q[:, -1, :]
    #         state['past_key_values'] = outputs.past_key_values
        
    #     return last_q

    def _forward_with_kvcache(self, input_ids: torch.Tensor) -> torch.Tensor:   # claude
        """Forward pass using the current model's KV cache."""
        model_idx = self._current_model_idx
        model = self._models[model_idx]
        state = self._model_states[model_idx]
        
        if state['past_key_values'] is None:
            outputs = model(input_ids)
            state['prob_history'] = outputs.logits[:, :, :self.vocab_size]
            for i in range(state['prob_history'].shape[-2]):
                state['prob_history'][:, i, :] = norm_logits(
                    state['prob_history'][:, i, :], 
                    self._temperature, 
                    self._top_k, 
                    self._top_p
                )
            # Store past_key_values in model's native format
            state['past_key_values'] = outputs.past_key_values
            last_q = state['prob_history'][:, -1, :]
        else:
            # Get cached length - needs to be adapted for different model types
            if hasattr(state['past_key_values'], 'get_seq_length'):
                # For newer models that use KeyValueCaches object
                cached_len = state['past_key_values'].get_seq_length()
            else:
                # For older models that use list of tuples
                cached_len = state['past_key_values'][0][0].shape[2]
                
            last_input_id = input_ids[:, cached_len:]
            outputs = model(
                last_input_id, 
                past_key_values=state['past_key_values'], 
                use_cache=True
            )
            not_cached_q = outputs.logits[:, :, :self.vocab_size]
            for i in range(not_cached_q.shape[-2]):
                not_cached_q[:, i, :] = norm_logits(
                    not_cached_q[:, i, :], 
                    self._temperature, 
                    self._top_k, 
                    self._top_p
                )
            state['prob_history'] = torch.cat(
                [state['prob_history'], not_cached_q], 
                dim=1
            ) if state['prob_history'] is not None else not_cached_q
            last_q = not_cached_q[:, -1, :]
            state['past_key_values'] = outputs.past_key_values
        
        return last_q

    def _generate_with_kvcache(self, prefix: torch.Tensor, gamma: int) -> torch.Tensor:
        """Generate γ tokens using all models and merge the best tokens."""
        all_sequences = []
        all_probs = []
        prefix_length = prefix.shape[1]

        # Generate sequences for each model
        for model_idx in range(len(self._models)):
            # print(f'Switching to model {model_idx}')
            self._switch_to_model(model_idx)
            # Reset model state for new prefix
            self._model_states[model_idx] = {'past_key_values': None, 'prob_history': None}
            model = self._models[model_idx]
            x = prefix.to(model.device)
            current_sequence = x.clone()
            
            for _ in range(gamma):
                q = self._forward_with_kvcache(current_sequence)
                next_tok = sample(q)  # Shape: (batch_size, 1)
                current_sequence = torch.cat((current_sequence, next_tok), dim=1)
            
            all_sequences.append(current_sequence)
            all_probs.append(self._model_states[model_idx]['prob_history'])

        # Merge sequences and probabilities
        merged_sequence = prefix.clone().to(self._models[0].device)
        merged_probs = []
        # print(f'Merging sequences {merged_sequence.shape}')
        
        for i in range(gamma):
            best_prob = -float('inf')
            best_slice = None
            best_model_idx = 0
            
            for model_idx in range(len(self._models)):
                pos = prefix_length + i
                if all_probs[model_idx] is None or pos >= all_probs[model_idx].shape[1]:
                    continue
                prob_slice = all_probs[model_idx][:, pos:pos+1, :]  # (batch, 1, vocab)
                
                if prob_slice.numel() == 0:
                    continue
                current_max = prob_slice.max(dim=-1)[0].max().item()
                if current_max > best_prob:
                    best_prob = current_max
                    best_slice = prob_slice
                    best_model_idx = model_idx
            
            if best_slice is not None:
                merged_probs.append(best_slice)
                token = all_sequences[best_model_idx][:, pos:pos+1]
                merged_sequence = torch.cat((merged_sequence, token), dim=1)
            else:
                # Fallback: use first model's prediction if all models fail
                token = all_sequences[0][:, prefix_length + i:prefix_length + i + 1]
                merged_sequence = torch.cat((merged_sequence, token), dim=1)
                merged_probs.append(all_probs[0][:, prefix_length + i:prefix_length + i + 1, :])

        # Aggregate prob_history
        if merged_probs:
            # print(f'Aggregating prob_history')
            merged_probs = torch.cat(merged_probs, dim=1)
            prefix_probs = all_probs[0][:, :prefix_length, :]
            self._prob_history = torch.cat([prefix_probs, merged_probs], dim=1)
        else:
            self._prob_history = all_probs[0][:, :prefix_length, :]

        return merged_sequence


    @torch.no_grad()
    def generate(self, input: torch.Tensor, gamma: int) -> torch.Tensor:
        return self._generate_with_kvcache(input, gamma)

    # @torch.no_grad()
    # def rollback(self, end_pos: int):   # Deep seek 
    #     """Rollback all models' states to a previous position."""
    #     for state in self._model_states:
    #         if state['past_key_values'] is not None:
    #             # Crop KV cache
    #             state['past_key_values'] = [
    #                 (k[:, :, :end_pos, :], v[:, :, :end_pos, :]) 
    #                 for k, v in state['past_key_values']
    #             ]
    #         if state['prob_history'] is not None:
    #             state['prob_history'] = state['prob_history'][:, :end_pos, :]
    #     # Also update aggregated prob_history
    #     if self._prob_history is not None:
    #         self._prob_history = self._prob_history[:, :end_pos, :]

    @torch.no_grad()
    def rollback(self, end_pos: int):
        """Rollback all models' states to a previous position."""
        for state in self._model_states:
            # print(f'Rolling back model state {state}')
            if state['past_key_values'] is not None:
                # Check the type of past_key_values
                if hasattr(state['past_key_values'], 'get_seq_length'):
                    # For newer models with KeyValueCaches object
                    # This will depend on what methods the specific object provides
                    # You might need to adapt this to the specific model being used
                    # For example, some models might have a method like:
                    if hasattr(state['past_key_values'], 'slice_and_return'):
                        state['past_key_values'] = state['past_key_values'].slice_and_return(0, end_pos)
                else:
                    # For older models with list of tuples
                    state['past_key_values'] = [
                        (k[:, :, :end_pos, :], v[:, :, :end_pos, :]) 
                        for k, v in state['past_key_values']
                    ]
            
            if state['prob_history'] is not None:
                state['prob_history'] = state['prob_history'][:, :end_pos, :]
        
        # Also update aggregated prob_history
        if self._prob_history is not None:
            self._prob_history = self._prob_history[:, :end_pos, :]