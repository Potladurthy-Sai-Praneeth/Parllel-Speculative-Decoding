import torch
import math
from .util import norm_logits, sample
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


'''
class KVCacheModel:   # Working code but slow
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

    def _forward_with_kvcache(self, input_ids: torch.Tensor) -> torch.Tensor:   
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
        """Generate tokens using all models and merge the best tokens."""
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


    @torch.no_grad()
    def rollback(self, end_pos: int): # claude 
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

'''
class KVCacheModel:   
    def __init__(self, models: list, temperature: float = 1, top_k: int = 0, top_p: float = 0) -> None:
        self._models = models
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self.vocab_size = None
        self._prob_history = None
        self._current_model_idx = 0  # Initialize to avoid potential errors
        
        self._model_states = [
            {'past_key_values': None, 'prob_history': None} 
            for _ in models
        ]

    def _switch_to_model(self, model_idx):
        """Switch active model without resetting state."""
        if model_idx >= len(self._models):
            model_idx = 0
        self._current_model_idx = model_idx

    def _forward_with_kvcache(self, input_ids: torch.Tensor) -> torch.Tensor:   
        """Forward pass using the current model's KV cache."""
        model_idx = self._current_model_idx
        model = self._models[model_idx]
        state = self._model_states[model_idx]
        
        # Ensure input is on the correct device - do this once
        input_ids = input_ids.to(model.device)
        
        if state['past_key_values'] is None:
            # Initial forward pass without KV cache
            outputs = model(input_ids)
            logits = outputs.logits[:, :, :self.vocab_size]
            
            # Apply normalization in a single batch operation if possible
            state['prob_history'] = logits.clone()
            for i in range(logits.shape[1]):
                state['prob_history'][:, i, :] = norm_logits(
                    logits[:, i, :], 
                    self._temperature, 
                    self._top_k, 
                    self._top_p
                )
                
            state['past_key_values'] = outputs.past_key_values
            last_q = state['prob_history'][:, -1, :]
        else:
            # Get cached length efficiently
            if hasattr(state['past_key_values'], 'get_seq_length'):
                cached_len = state['past_key_values'].get_seq_length()
            else:
                cached_len = state['past_key_values'][0][0].shape[2]
                
            # Process only new tokens (optimization for incremental processing)
            last_input_id = input_ids[:, cached_len:]
            
            # Skip inference if no new tokens
            if last_input_id.shape[1] == 0:
                return state['prob_history'][:, -1, :]
                
            outputs = model(
                last_input_id, 
                past_key_values=state['past_key_values'], 
                use_cache=True
            )
            
            not_cached_q = outputs.logits[:, :, :self.vocab_size]
            
            # Apply normalization
            for i in range(not_cached_q.shape[1]):
                not_cached_q[:, i, :] = norm_logits(
                    not_cached_q[:, i, :], 
                    self._temperature, 
                    self._top_k, 
                    self._top_p
                )
                
            # Update probability history efficiently
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
        first_device = self._models[0].device
        model_count = len(self._models)

        # Generate sequences for each model
        for model_idx in range(model_count):
            self._switch_to_model(model_idx)
            # Reset model state for new prefix
            self._model_states[model_idx] = {'past_key_values': None, 'prob_history': None}
            model = self._models[model_idx]
            device = model.device
            
            # Move prefix to model device once
            x = prefix.to(device)
            current_sequence = x.clone()
            
            # Generate tokens
            for _ in range(gamma):
                q = self._forward_with_kvcache(current_sequence)
                next_tok = sample(q)  # Shape: (batch_size, 1)
                current_sequence = torch.cat((current_sequence, next_tok), dim=1)
            
            # Store results
            all_sequences.append(current_sequence)
            all_probs.append(self._model_states[model_idx]['prob_history'])

        # Create merged sequence on the first device
        merged_sequence = prefix.clone().to(first_device)
        merged_probs = []
        
        # Process positions one by one
        for i in range(gamma):
            pos = prefix_length + i
            best_prob = -float('inf')
            best_model_idx = 0
            best_token = None
            best_slice = None
            
            # Find best token across models
            for model_idx in range(model_count):
                probs = all_probs[model_idx]
                if probs is None or pos >= probs.shape[1]:
                    continue
                    
                prob_slice = probs[:, pos:pos+1, :]
                if prob_slice.numel() == 0:
                    continue
                    
                # Get maximum probability
                current_max = prob_slice.max(dim=-1)[0].max().item()
                if current_max > best_prob:
                    best_prob = current_max
                    best_slice = prob_slice
                    best_model_idx = model_idx
                    # Get token directly from sequence to avoid recomputing
                    best_token = all_sequences[best_model_idx][:, pos:pos+1].to(first_device)
            
            # Fall back to first model if no valid token was found
            if best_token is None:
                best_token = all_sequences[0][:, pos:pos+1].to(first_device)
                best_slice = all_probs[0][:, pos:pos+1, :]
                
            # Add to merged results
            merged_sequence = torch.cat((merged_sequence, best_token), dim=1)
            merged_probs.append(best_slice.to(first_device))  # Move to first device

        # Aggregate probability history
        if merged_probs:
            # Concatenate merged probabilities
            merged_probs = torch.cat(merged_probs, dim=1)
            prefix_probs = all_probs[0][:, :prefix_length, :].to(first_device)
            self._prob_history = torch.cat([prefix_probs, merged_probs], dim=1)
        else:
            self._prob_history = all_probs[0][:, :prefix_length, :].to(first_device)

        return merged_sequence

    @torch.no_grad()
    def generate(self, input: torch.Tensor, gamma: int) -> torch.Tensor:
        # Use no_grad context for entire function to avoid building computational graph
        return self._generate_with_kvcache(input, gamma)

    @torch.no_grad()
    def rollback(self, end_pos: int):
        """Rollback all models' states to a previous position."""
        for state in self._model_states:
            if state['past_key_values'] is not None:
                if hasattr(state['past_key_values'], 'get_seq_length'):
                    if hasattr(state['past_key_values'], 'slice_and_return'):
                        state['past_key_values'] = state['past_key_values'].slice_and_return(0, end_pos)
                else:
                    state['past_key_values'] = [
                        (k[:, :, :end_pos, :], v[:, :, :end_pos, :]) 
                        for k, v in state['past_key_values']
                    ]
            
            if state['prob_history'] is not None:
                state['prob_history'] = state['prob_history'][:, :end_pos, :]

        if self._prob_history is not None:
            self._prob_history = self._prob_history[:, :end_pos, :]