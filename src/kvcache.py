# import torch
# from .util import norm_logits, sample
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_mem_efficient_sdp(False)


# class KVCacheModel():
#     def __init__(self, model : torch.nn.Module, temperature : float = 1, top_k : int = 0, top_p : float = 0) -> None:
#         self._model = model
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
            
#             if isinstance(outputs.past_key_values, tuple):
#                 self._past_key_values = outputs.past_key_values[0]
#             else:
#                 self._past_key_values = outputs.past_key_values

#             last_q = self._prob_history[:, -1, :]
#         else:
#             # return the last token's logits
#             if isinstance(self._past_key_values, tuple):
#                 self._past_key_values = self._past_key_values[0]

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
#             # self._past_key_values = outputs.past_key_values
#             if isinstance(outputs.past_key_values, tuple):
#                 self._past_key_values = outputs.past_key_values[0]
#             else:
#                 self._past_key_values = outputs.past_key_values
        
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
#         x = prefix

#         for _ in range(gamma):
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
#         self._past_key_values.crop(end_pos)
#         self._prob_history = self._prob_history[:, :end_pos, :]





import torch
from .util import norm_logits, sample
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)


class KVCacheModel():
    def __init__(self, model : torch.nn.Module, temperature : float = 1, top_k : int = 0, top_p : float = 0) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None
        self._seq_len = 0  # Track sequence length manually

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        
        # Set attribute to fix AttributeError
        self.vocab_size = getattr(model.config, "vocab_size", 32000)  # Default fallback

    def _forward_with_kvcache(self, input_ids : torch.Tensor) -> torch.Tensor:
        if self._past_key_values is None:
            # First forward pass without past_key_values
            outputs = self._model(input_ids)
            self._prob_history = outputs.logits[:, :, :self.vocab_size]
            for i in range(self._prob_history.shape[-2]):   
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
            
            self._past_key_values = outputs.past_key_values
            self._seq_len = input_ids.shape[1]  # Initialize sequence length

            last_q = self._prob_history[:, -1, :]
        else:
            # Handle different past_key_values formats
            cached_len = self._seq_len
                
            # Get only the new token(s)
            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            
            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            
            not_cached_q = outputs.logits[:, :, :self.vocab_size]
            
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
            for i in range(not_cached_q.shape[-2]):   
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
                
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values
            self._seq_len += last_input_id.shape[1]  # Update sequence length
        
        return last_q


    def _generate_with_kvcache(self, prefix : torch.Tensor, 
                                    gamma : int) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix

        for _ in range(gamma):
            q = self._forward_with_kvcache(x)
            next_tok = sample(q)
            x = torch.cat((x, next_tok), dim=1)
        return x

    @torch.no_grad()
    def generate(self, input : torch.Tensor, gamma : int) -> torch.Tensor:
        output = self._generate_with_kvcache(input, gamma)
        return output
    
    @torch.no_grad()
    def rollback(self, end_pos : int):
        # Handle different past_key_value formats for rollback
        if hasattr(self._past_key_values, "crop"):
            # If it has a crop method, use it
            self._past_key_values.crop(end_pos)
        else:
            # For tensor-based past_key_values, we'll need to reset and regenerate
            # This is a fallback that might not be optimal
            self._past_key_values = None
            self._seq_len = 0
            
        # Update prob history in any case
        if end_pos < self._prob_history.shape[1]:
            self._prob_history = self._prob_history[:, :end_pos, :]
            self._seq_len = end_pos