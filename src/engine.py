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
        self.color_print(f"Loading models: \n Draft : {self.args.draft_models}\n Target : {self.args.target_model}", 3)
       
        if self.args.eval_mode == "para_sd":
            if self.accelerator.is_main_process:
                for idx in range(self.args.num_samples_per_task):
                    self.all_draft_models.append(AutoModelForCausalLM.from_pretrained(self.args.draft_models[idx], device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True).eval())
            else:
                self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model, device_map="balanced_low_0", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        

    def load_tokenizer(self):
        # * load tokenizers
        self.color_print(f"Loading tokenizer of {self.args.target_model}...", 3)
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

    
    # @torch.no_grad()
    # def parallel_speculative_decoding(self, prefix):
    #     # ... (initial setup remains the same)
    #     if self.accelerator.is_main_process:
    #         for idx ,m in enumerate(self.all_draft_models):
    #             self.kv_cache_models[idx] = KVCacheModel(m, self.args.temp, self.args.top_k, self.args.top_p)
    #             self.kv_cache_models[idx].vocab_size = self.vocab_size
    #         device = self.all_draft_models[-1].device
    #     else:
    #         model = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
    #         model.vocab_size = self.vocab_size
    #         device = self.target_model.device

    #     max_tokens = prefix.shape[1] + self.args.max_tokens
        
    #     # this flag is used to determine the current verify mode.
    #     cur_mode = True
    #     num_acc_token = 0

    #     print(f'Going inside While loop')

    #     while prefix.shape[1] < max_tokens:
    #         prefix_len = prefix.shape[1]
    #         input_ids = prefix.to(device)

    #         # Main process: Generate all draft probabilities
    #         if self.accelerator.is_main_process:
    #             draft_probs = []
    #             for idx, kv_model in self.kv_cache_models.items():
    #                 x = kv_model.generate(input_ids, self.args.gamma)
    #                 prob = kv_model._prob_history[:, prefix_len - self.args.gamma - 1:prefix_len, :self.vocab_size].to(torch.float32)
    #                 # prob[:, 0, 0] = -1
    #                 # prob[:, 0, 1:self.args.gamma*2] = x[:, prefix_len-self.args.gamma+1:prefix_len+self.args.gamma]
    #                 draft_probs.append(prob)
    #                 self.draft_forward_times += self.args.gamma
    #                 print(f'Finihsed generation of draft:{idx} with shape {prob.shape}')
    #             # Stack all draft probs into a batch [num_drafts, batch, gamma, vocab]
    #             draft_probs = torch.stack(draft_probs, dim=0)
    #         else:
    #             # Non-main process: Generate target probability once
    #             x = model.generate(input_ids, 1)
    #             prob = model._prob_history[:, prefix_len - self.args.gamma - 1:prefix_len, :self.vocab_size].to(torch.float32)
    #             prob = prob.squeeze(0).to("cuda:1")
    #             self.target_forward_times += 1
    #             print(f'Finihsed generation of target with shape {prob.shape}')

    #         # Gather all probabilities across processes
    #         self.accelerator.wait_for_everyone()
    #         gathered_probs = self.accelerator.gather(draft_probs if self.accelerator.is_main_process else prob)

    #         # Split into draft and target probabilities
    #         # if self.accelerator.is_main_process:
    #         num_drafts = len(self.kv_cache_models)
    #         all_draft_probs = gathered_probs[:num_drafts]  # [num_drafts, 1, gamma, vocab]
    #         target_probs = gathered_probs[num_drafts:]     # [1, 1, gamma, vocab]

    #         # Track the best candidate across all drafts
    #         print(f'All draft probs shape is {all_draft_probs.shape}')
    #         print(f'Target probs shape is {target_probs.shape}')
    #         temp_prefix = prefix.clone()
           
    #         temp_tokens = 0
    #         num_accept_tokens= []

    #         # Verify each draft against the target
    #         print(f'Going inside for loop for comparison')
    #         for draft_idx in range(all_draft_probs.shape[0]):
    #             print(f'Comapring the probs of draft with index {draft_idx}')
    #             auxilairy_prefix = prefix.clone()
    #             draft_prob_single = all_draft_probs[draft_idx]
    #             draft_ids = draft_prob_single[:, 0, 1:self.args.gamma * 2].int()
    #             draft_prob = draft_prob_single[:, 1:, :]
    #             target_prob = target_probs[:, 1:, :]

    #             # Original verification logic for one draft-target pair
    #             # ... (reuse the existing code for token acceptance/rejection)
    #             # Update best_prefix if this draft has more accepted tokens

    #             if cur_mode:
    #                 first_token = draft_ids[:, -self.args.gamma]
    #                 torch.manual_seed(self.seed + prefix_len)

    #                 r = torch.rand(1, device=device)
    #                 if  r > target_prob[:, -1, first_token] / draft_prob[:, -1, first_token]:
    #                     # reject the first token
    #                     t = sample(max_fn(target_prob[:, -1, :] - draft_prob[:, -1, :]))
    #                     auxilairy_prefix = torch.cat((input_ids, t), dim=1)
                        
    #                     # record the number of accepted tokens
    #                     num_accept_tokens.append(temp_tokens)
    #                     temp_tokens = 0
                        
    #                     if self.accelerator.is_main_process:
    #                         # rollback the small model kv cache
    #                         kv_model.rollback(prefix_len)
    #                 else:
    #                     # accept the first token, change the mode
    #                     cur_mode = False
    #                     auxilairy_prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma:]), dim=1)
    #                     temp_tokens += 1

    #             else:
    #                 n = self.args.gamma
    #                 for i in range(self.args.gamma):
    #                     token = draft_ids[:, i]
    #                     torch.manual_seed(self.seed + prefix_len - self.args.gamma + i)
    #                     r = torch.rand(1, device=device)
    #                     if r > target_prob[:, i, token] / draft_prob[:, i, token]:
    #                         n = i
    #                         break
    #                 if n == self.args.gamma:
    #                     # accept all guess tokens
    #                     auxilairy_prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma:]), dim=1)
    #                     temp_tokens += self.args.gamma
    #                 else:
    #                     # reject someone, change the mode
    #                     assert n < self.args.gamma
    #                     cur_mode = True
    #                     t = sample(max_fn(target_prob[:, n, :] - draft_prob[:, n, :]))
                        
    #                     auxilairy_prefix = torch.cat((input_ids[:, :prefix_len-self.args.gamma + n + 1], t), dim=1)
    #                     num_accept_tokens.append(temp_tokens + n)
    #                     temp_tokens = 0
    #                     # rollback both the large model and the small model kv cache
    #                     model.rollback(prefix_len - self.args.gamma +n+1)
    #                     kv_model.rollback(prefix_len - self.args.gamma +n+1)
                
    #             if len(num_accept_tokens)>=len(self.num_acc_tokens):
    #                 self.num_acc_tokens = num_accept_tokens
    #                 temp_prefix = auxilairy_prefix
    #                 num_acc_token = temp_tokens
                       
    #             temp_tokens = 0
    #             num_accept_tokens = []
    #             cur_mode = True

    #         prefix = temp_prefix.clone()
    #         print(f'After one round of comparison we have prefix shape is {prefix.shape}')

    #     return prefix
        
    
    @torch.no_grad()
    def parallel_speculative_decoding(self, prefix):   # My code
        # parallel speculative decoding  
        if self.accelerator.is_main_process:
            for idx ,m in enumerate(self.all_draft_models):
                print(f'Loading draft model {idx}')
                self.kv_cache_models[idx] = KVCacheModel(m, self.args.temp, self.args.top_k, self.args.top_p)
                self.kv_cache_models[idx].vocab_size = self.vocab_size
            device = self.all_draft_models[-1].device
        else:
            print(f'Loading target model')
            model = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = self.target_model.device

        max_tokens = prefix.shape[1] + self.args.max_tokens
        
        # this flag is used to determine the current verify mode.
        cur_mode = True
        num_acc_token = 0

        while prefix.shape[1] < max_tokens:

            prefix_len = prefix.shape[1]
            input_ids = prefix.to(device)

            temp_prefix = prefix.clone()
           
            temp_tokens = 0
            num_accept_tokens= []

            # if not self.accelerator.is_main_process:
            #     print(f'Target model is generating the probabilities')
            #     x = model.generate(input_ids, 1)
            #     prob = model._prob_history[:, prefix_len-self.args.gamma-1:prefix_len, :self.vocab_size].to(torch.float32)
            #     prob = prob.to("cuda:1")
            #     self.target_forward_times += 1
            print(f'kv cache models are {self.kv_cache_models}')
            if self.kv_cache_models == {}:
                break
            for idx,kv_model in self.kv_cache_models.items():
                auxilairy_prefix = prefix.clone()
                print(f'Draft model {idx} is generating the probabilities')
                if self.accelerator.is_main_process:
                    x = kv_model.generate(input_ids, self.args.gamma)
                    prob = kv_model._prob_history[:, prefix_len-self.args.gamma-1:prefix_len, :self.vocab_size].to(torch.float32)
                    prob[:, 0, 0] = -1
                    prob[:, 0, 1:self.args.gamma*2] = x[:, prefix_len-self.args.gamma+1:prefix_len+self.args.gamma]
                    self.draft_forward_times += self.args.gamma
                
                else:
                    print(f'Target model is generating the probabilities')
                    x = model.generate(input_ids, 1)
                    prob = model._prob_history[:, prefix_len-self.args.gamma-1:prefix_len, :self.vocab_size].to(torch.float32)
                    prob = prob.to("cuda:1")
                    self.target_forward_times += 1
                
                self.accelerator.wait_for_everyone()

                # verification
                all_prob = self.accelerator.gather(prob).to(device)

                draft_ids = all_prob[0, [0], 1:self.args.gamma*2].int()
                draft_prob = all_prob[[0], 1:, :]
                target_prob = all_prob[[1], 1:, :]

                print(f'All draft probs shape is {all_prob.shape}')
                print(f'Draft probs shape is {draft_prob.shape}')
                print(f'Target probs shape is {target_prob.shape}')

                print(f'Proceeding for verification')

                if cur_mode:
                    first_token = draft_ids[:, -self.args.gamma]
                    torch.manual_seed(self.seed + prefix_len)

                    r = torch.rand(1, device=device)
                    if  r > target_prob[:, -1, first_token] / draft_prob[:, -1, first_token]:
                        # reject the first token
                        t = sample(max_fn(target_prob[:, -1, :] - draft_prob[:, -1, :]))
                        auxilairy_prefix = torch.cat((input_ids, t), dim=1)
                        
                        # record the number of accepted tokens
                        num_accept_tokens.append(temp_tokens)
                        temp_tokens = 0
                        
                        if self.accelerator.is_main_process:
                            # rollback the small model kv cache
                            kv_model.rollback(prefix_len)
                    else:
                        # accept the first token, change the mode
                        cur_mode = False
                        auxilairy_prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma:]), dim=1)
                        temp_tokens += 1

                else:
                    n = self.args.gamma
                    for i in range(self.args.gamma):
                        token = draft_ids[:, i]
                        torch.manual_seed(self.seed + prefix_len - self.args.gamma + i)
                        r = torch.rand(1, device=device)
                        if r > target_prob[:, i, token] / draft_prob[:, i, token]:
                            n = i
                            break
                    if n == self.args.gamma:
                        # accept all guess tokens
                        auxilairy_prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma:]), dim=1)
                        temp_tokens += self.args.gamma
                    else:
                        # reject someone, change the mode
                        assert n < self.args.gamma
                        cur_mode = True
                        t = sample(max_fn(target_prob[:, n, :] - draft_prob[:, n, :]))
                        
                        auxilairy_prefix = torch.cat((input_ids[:, :prefix_len-self.args.gamma + n + 1], t), dim=1)
                        num_accept_tokens.append(temp_tokens + n)
                        temp_tokens = 0
                        # rollback both the large model and the small model kv cache
                        model.rollback(prefix_len - self.args.gamma +n+1)
                        kv_model.rollback(prefix_len - self.args.gamma +n+1)
                
                if len(num_accept_tokens)>=len(self.num_acc_tokens):
                    self.num_acc_tokens = num_accept_tokens
                    temp_prefix = auxilairy_prefix
                    num_acc_token = temp_tokens
                       
                temp_tokens = 0
                num_accept_tokens = []
                cur_mode = True

            prefix = temp_prefix.clone()
            # print(f'After one round of comparison we have prefix shape is {prefix.shape}')

        return prefix

    
    @abstractmethod
    def eval(self):
        pass

    def color_print(self, content: str, color_number: int=4):
        """print content with color. Some color numbers are listed: Gray: 0, Red: 1, Green: 2, Yellow: 3, Blue: 4."""
        if self.accelerator.is_main_process:
            print(f"\033[9{color_number}m{content}\033[0m")
