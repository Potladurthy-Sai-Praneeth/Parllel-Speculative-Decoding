import torch
import transformers
import warnings
transformers.utils.logging.set_verbosity(40)
warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod
from accelerate import Accelerator
from .kvcache import KVCacheModel
# from .kvcache4RC import KVCacheModel as KVCache2Model
from .util import seed_everything, norm_logits, sample, max_fn
import time
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


class Decoding(ABC):
    def __init__(self, args):
        self.args = args
        self.accelerator = Accelerator()
        
        seed_everything(self.args.seed)
        self.seed = self.args.seed
        self.seed_set = set()
        
        # ! only parallel speculative decoding can use 2 processes
        # record metrics for report
        self.draft_forward_times = 0
        self.target_forward_times = 0
        self.num_acc_tokens = []

        self.all_draft_models = []
        self.kv_cache_models = {}

        self.vocab_size = self.args.vocab_size
    
    def load_model(self):
        # * load models according to different evaluation methods.
        self.color_print(f"Loading models: \n Draft : {self.args.draft_models}\n Target : {self.args.target_model}")
       
        if self.args.eval_mode == "para_sd":
            if self.accelerator.is_main_process:
                for idx in range(self.args.num_samples_per_task):
                    self.all_draft_models.append(AutoModelForCausalLM.from_pretrained(self.args.draft_models[idx], device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True).eval())
            else:
                self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model, device_map="balanced_low_0", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        

    def load_tokenizer(self):
        # * load tokenizers
        self.color_print(f"Loading tokenizer of {self.args.target_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.target_model, trust_remote_code=True)
        self.tokenizer.padding_side = "right"

    @abstractmethod
    def load_data(self):
        pass
    
    @abstractmethod
    def preprocess(self, input_text):
        pass
    
    @abstractmethod
    def postprocess(self, input_text, output_text):
        pass

    @torch.no_grad()
    def parallel_speculative_decoding(self, prefix):
        # Initialize models with KV cache
        if self.accelerator.is_main_process:
            for idx, m in enumerate(self.all_draft_models):
                self.kv_cache_models[idx] = KVCacheModel(m, self.args.temp, self.args.top_k, self.args.top_p)
                self.kv_cache_models[idx].vocab_size = self.vocab_size
            device = self.all_draft_models[0].device  # Use first draft model's device
        else:
            model = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = self.target_model.device

        max_tokens = prefix.shape[1] + self.args.max_tokens
        
        # Synchronize prefix across processes at the start
        prefix = self.accelerator.gather(prefix.to(device))
        print(prefix.shape)
        # prefix = gathered_prefix[0].clone()  # Use first copy since all should be identical
        
        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]
            input_ids = prefix.to(device)
            
            # Generate draft and target probabilities
            if self.accelerator.is_main_process:
                # Main process: Generate draft probabilities
                draft_probs = []
                draft_tokens = []
                
                for idx, kv_model in self.kv_cache_models.items():
                    # Generate gamma tokens ahead
                    x = kv_model.generate(input_ids, self.args.gamma)
                    prob = kv_model._prob_history[:, prefix_len - self.args.gamma - 1:prefix_len, :self.vocab_size].to(torch.float32)
                    tokens = x[:, prefix_len:prefix_len+self.args.gamma]
                    
                    draft_probs.append(prob)
                    draft_tokens.append(tokens)
                    self.draft_forward_times += self.args.gamma
                
                # Stack all draft probabilities
                draft_probs = torch.stack(draft_probs, dim=0)  # [num_drafts, batch, seq_len, vocab]
                draft_tokens = torch.stack(draft_tokens, dim=0)  # [num_drafts, batch, gamma]
            else:
                # Non-main process: Generate target probability
                x = model.generate(input_ids, 1)
                target_prob = model._prob_history[:, prefix_len - self.args.gamma - 1:prefix_len, :self.vocab_size].to(torch.float32)
                target_prob = target_prob.unsqueeze(0)  # [1, batch, seq_len, vocab]
                self.target_forward_times += 1
            
            # Synchronize across processes before gathering
            self.accelerator.wait_for_everyone()
            
            # Each process creates a placeholder tensor for the other process
            if self.accelerator.is_main_process:
                # Main process has draft_probs but needs a placeholder for target_prob
                placeholder_target = torch.zeros_like(draft_probs[0:1])  # Create placeholder with same shape as one draft
                gathered_probs = self.accelerator.gather_object([draft_probs, placeholder_target])
            else:
                # Non-main process has target_prob but needs placeholder for draft_probs
                gathered_probs = self.accelerator.gather_object([target_prob])
            
            # Both processes now have the complete list of tensors
            # We know the main process provided draft probs first, followed by placeholder
            # The non-main process provided the target prob
            
            # Only do verification in main process to avoid inconsistency
            best_prefix = prefix.clone()
            best_total_accepted = 0
            
            # Define shapes for consistency
            num_drafts = len(self.kv_cache_models) if hasattr(self, 'kv_cache_models') else 0
            
            if self.accelerator.is_main_process:
                # Extract probabilities from gathered objects
                all_draft_probs = gathered_probs[0]  # This is what the main process provided
                target_prob = gathered_probs[-1]  # Last item is from non-main process
                
                # Evaluate each draft model
                for draft_idx in range(num_drafts):
                    # Get this draft's probabilities and tokens
                    draft_prob = all_draft_probs[draft_idx]
                    draft_ids = draft_tokens[draft_idx]  # We already have this in main process
                    
                    # Initialize for token verification
                    cur_mode = True  # Start in first-token verification mode
                    temp_prefix = prefix.clone()
                    total_accepted = 0
                    
                    # First-token verification
                    if cur_mode:
                        first_token = draft_ids[:, 0]
                        torch.manual_seed(self.seed + prefix_len)
                        r = torch.rand(1, device=device)
                        
                        # Check acceptance of first token
                        if r <= target_prob[:, -1, first_token] / draft_prob[:, -1, first_token]:
                            # Accept first token, update temp_prefix
                            temp_prefix = torch.cat((input_ids, first_token.unsqueeze(1)), dim=1)
                            total_accepted += 1
                            
                            # Switch to multi-token mode for remaining tokens
                            cur_mode = False
                            
                            # Verify remaining tokens (if any)
                            if self.args.gamma > 1:
                                n = self.args.gamma
                                for i in range(1, self.args.gamma):
                                    token = draft_ids[:, i]
                                    torch.manual_seed(self.seed + prefix_len + i)
                                    r = torch.rand(1, device=device)
                                    
                                    if r > target_prob[:, i, token] / draft_prob[:, i, token]:
                                        n = i
                                        break
                                
                                if n == self.args.gamma:
                                    # All remaining tokens accepted
                                    temp_prefix = torch.cat((input_ids, draft_ids[:, :self.args.gamma]), dim=1)
                                    total_accepted = self.args.gamma
                                else:
                                    # Some token rejected, sample new token
                                    t = sample(max_fn(target_prob[:, n, :] - draft_prob[:, n, :]))
                                    temp_prefix = torch.cat((temp_prefix, t), dim=1)
                                    total_accepted = n + 1  # Count first token + accepted tokens
                                    
                                    # Rollback KV caches
                                    for _, kv_model in self.kv_cache_models.items():
                                        kv_model.rollback(prefix_len + n + 1)
                        else:
                            # Reject first token, sample from target
                            t = sample(max_fn(target_prob[:, -1, :] - draft_prob[:, -1, :]))
                            temp_prefix = torch.cat((input_ids, t), dim=1)
                            total_accepted = 1  # Just the sampled token
                            
                            # Rollback draft KV caches
                            for _, kv_model in self.kv_cache_models.items():
                                kv_model.rollback(prefix_len)
                    
                    # Check if this draft produced better results
                    if total_accepted > best_total_accepted:
                        best_total_accepted = total_accepted
                        best_prefix = temp_prefix.clone()
                
                # Update with the best prefix found
                prefix = best_prefix.clone()
            
            # Broadcast the best prefix from main process to all others
            if self.accelerator.is_main_process:
                # Main process has the best prefix
                best_prefix_to_broadcast = prefix.clone()
            else:
                # Non-main process creates a placeholder
                best_prefix_to_broadcast = None
                
            # Broadcast and receive the best prefix
            prefix = self.accelerator.broadcast_object_list(
                [best_prefix_to_broadcast] if self.accelerator.is_main_process else [None]
            )[0]
            
            # Print current generation progress
            if self.accelerator.is_main_process:
                print(f"Generated {prefix.shape[1] - self.args.prompt_len} tokens")
        
        return prefix
    
    @abstractmethod
    def eval(self):
        pass

    def color_print(self, content: str, color_number: int=4):
        """print content with color. Some color numbers are listed: Gray: 0, Red: 1, Green: 2, Yellow: 3, Blue: 4."""
        if self.accelerator.is_main_process:
            print(f"\033[9{color_number}m{content}\033[0m")