import torch
import transformers
import warnings
transformers.utils.logging.set_verbosity(40)
warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod
from accelerate import Accelerator
from .kvcache import KVCacheModel
from .kvcache4RC import KVCacheModel as KVCache2Model
from .util import seed_everything, norm_logits, sample, max_fn
import time


class Decoding(ABC):
    def __init__(self, args):
        self.args = args
        self.accelerator = Accelerator()
        
        seed_everything(self.args.seed)
        self.seed = self.args.seed
        self.seed_set = set()
        
        # # ! only parallel speculative decoding can use 2 processes
        # assert (self.accelerator.num_processes == 1 and args.eval_mode in ["small", "large", "sd"]) or (self.accelerator.num_processes == 2 and args.eval_mode in ["para_sd", "para_sd_wo_1", "para_sd_wo_1", "rc_para_sd"])

        # record metrics for report
        self.draft_forward_times = 0
        self.target_forward_times = 0
        self.num_acc_tokens = []
        self.model_counter = 0
        self.all_draft_models ={}
        self.all_kv_cache_models = {}
    
    def load_model(self):
        # * load models according to different evaluation methods.
        self.color_print(f"Loading models:\n{self.args.draft_model_1}\n{self.args.target_model}", 3)

        if self.args.eval_mode == "para_sd":
            if self.accelerator.is_main_process:
                self.draft_model_1 = AutoModelForCausalLM.from_pretrained(self.args.draft_model_1, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
                self.all_draft_models[self.model_counter] = self.draft_model_1
                self.model_counter+=1
                if self.args.draft_model_2 is not None:
                    self.draft_model_2 = AutoModelForCausalLM.from_pretrained(self.args.draft_model_2, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
                    self.all_draft_models[self.model_counter] = self.draft_model_2
                    self.model_counter+=1
                if self.args.draft_model_3 is not None:
                    self.draft_model_3 = AutoModelForCausalLM.from_pretrained(self.args.draft_model_3, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
                    self.all_draft_models[self.model_counter] = self.draft_model_3
                    self.model_counter+=1
            else:
                # self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model, device_map="balanced_low_0", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
                self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model, device_map="cuda:1", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        self.vocab_size = self.args.vocab_size

    def load_tokenizer(self):
        # * load tokenizers
        self.color_print(f"Loading tokenizer of {self.args.draft_model_1}...", 3)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.draft_model_1, trust_remote_code=True)
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
        # parallel speculative decoding
        if self.accelerator.is_main_process:
            for idx,value in self.all_draft_models.items():
                self.all_kv_cache_models[idx] = KVCache2Model(value, self.args.temp, self.args.top_k, self.args.top_p)
                self.all_kv_cache_models[idx].vocab_size = self.vocab_size
            device = self.all_kv_cache_models[0].device

        else:
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
            if self.accelerator.is_main_process:
                all_draft_probs = []
                for idx, kv_model in self.all_kv_cache_models.items():
                    x = kv_model.generate(input_ids, self.args.gamma)
                    prob = kv_model._prob_history[:, prefix_len-self.args.gamma-1:prefix_len, :self.vocab_size].to(torch.float32)
                    prob[:, 0, 0] = -1
                    prob[:, 0, 1:self.args.gamma*2] = x[:, prefix_len-self.args.gamma+1:prefix_len+self.args.gamma]
                    self.draft_forward_times += self.args.gamma
                    all_draft_probs.append(prob)

                prob = torch.stack(all_draft_probs, dim=0)
            else:
                x = model.generate(input_ids, 1)
                prob = model._prob_history[:, prefix_len-self.args.gamma-1:prefix_len, :self.vocab_size].to(torch.float32)
                prob = prob.to("cuda:1")
                self.target_forward_times += 1
            
            self.accelerator.wait_for_everyone()

            # At this point we have probs of multiple draft models stacked in prob thus we need to modify the following for batch comparison

            # verification
            all_prob = self.accelerator.gather(prob).to(device)

            prefix_copy = prefix.clone()
            intermediate_prefix = None

            accept_token_copy = self.num_acc_tokens.copy()
            intermediate_acc_tokens = []
            intermediate_num_tokens = 0

            for idx,kv_model in self.all_kv_cache_models.items():
                
                draft_ids = all_prob[0, [idx], 1:self.args.gamma*2].int()
                draft_prob = all_prob[[idx], 1:, :]
                target_prob = all_prob[[-1], 1:, :]


                if cur_mode:
                    first_token = draft_ids[:, -self.args.gamma]
                    torch.manual_seed(self.seed + prefix_len)

                    r = torch.rand(1, device=device)
                    if  r > target_prob[:, -1, first_token] / draft_prob[:, -1, first_token]:
                        # reject the first token
                        t = sample(max_fn(target_prob[:, -1, :] - draft_prob[:, -1, :]))
                        intermediate_prefix = torch.cat((input_ids, t), dim=1)
                        
                        # record the number of accepted tokens
                        intermediate_acc_tokens.append(intermediate_num_tokens)
                        intermediate_num_tokens = 0
                        
                        if self.accelerator.is_main_process:
                            # rollback the small model kv cache
                            kv_model.rollback(prefix_len)
                    else:
                        # accept the first token, change the mode
                        cur_mode = False
                        intermediate_prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma:]), dim=1)
                        intermediate_num_tokens += 1

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
                        intermediate_prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma:]), dim=1)
                        intermediate_num_tokens += self.args.gamma
                    else:
                        # reject someone, change the mode
                        assert n < self.args.gamma
                        cur_mode = True
                        t = sample(max_fn(target_prob[:, n, :] - draft_prob[:, n, :]))
                        
                        intermediate_prefix = torch.cat((input_ids[:, :prefix_len-self.args.gamma + n + 1], t), dim=1)
                        intermediate_num_tokens.append(intermediate_num_tokens + n)
                        intermediate_num_tokens = 0
                        # rollback both the large model and the small model kv cache
                        kv_model.rollback(prefix_len - self.args.gamma +n+1)
                        
                if intermediate_prefix.shape[1] >= prefix_copy.shape[1]:
                    prefix_copy = intermediate_prefix
                    intermediate_prefix = None
                else:
                    intermediate_prefix = None

                if len(intermediate_acc_tokens)>=accept_token_copy:
                    accept_token_copy = intermediate_acc_tokens
                    intermediate_acc_tokens = []
                    intermediate_num_tokens = 0
                else:
                    intermediate_acc_tokens = []
                    intermediate_num_tokens = 0

            prefix = prefix_copy
            self.num_acc_tokens.extend(accept_token_copy)
                
        return prefix

    
    @abstractmethod
    def eval(self):
        pass

    def color_print(self, content: str, color_number: int=4):
        """print content with color. Some color numbers are listed: Gray: 0, Red: 1, Green: 2, Yellow: 3, Blue: 4."""
        if self.accelerator.is_main_process:
            print(f"\033[9{color_number}m{content}\033[0m")
