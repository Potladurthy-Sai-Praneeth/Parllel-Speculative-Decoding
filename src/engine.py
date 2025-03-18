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
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

class Decoding(ABC):
    def __init__(self, args):
        self.args = args
        self.accelerator = Accelerator()
        
        seed_everything(self.args.seed)
        self.seed = self.args.seed
        self.seed_set = set()
        
        # record metrics for report
        self.draft_forward_times = 0
        self.target_forward_times = 0
        self.num_acc_tokens = []

        self.all_draft_models = []

        self.vocab_size = self.args.vocab_size
    
    def load_model(self):
        # * load models according to different evaluation methods.
        # self.color_print(f"Loading models: \n Draft : {self.args.draft_models}\n Target : {self.args.target_model}", 3)
       
        if self.args.eval_mode == "para_sd":
            if self.accelerator.is_main_process:
                for idx in range(self.args.num_samples_per_task):
                    self.all_draft_models.append(AutoModelForCausalLM.from_pretrained(self.args.draft_models[idx], device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True).eval())
            else:
                self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        
        

    def load_tokenizer(self):
        # * load tokenizers
        # self.color_print(f"Loading tokenizer of {self.args.target_model}...", 3)
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
        # parallel speculative decoding
        if self.accelerator.is_main_process:
            model = KVCache2Model(self.all_draft_models, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = torch.device("cuda:0")
        else:
            model = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = torch.device("cuda:1")

        max_tokens = prefix.shape[1] + self.args.max_tokens
        
        # this flag is used to determine the current verify mode.
        cur_mode = True
        num_acc_token = 0

        # print(f'Going inside while loop')
        while prefix.shape[1] < max_tokens:
            # print(f'Beginning Prefix shape is {prefix.shape}')
            prefix_len = prefix.shape[1]
            
            input_ids = prefix.to(device)
            if self.accelerator.is_main_process:
                # print(f'Generating the draft model outputs')
                x = model.generate(input_ids, self.args.gamma)
                prob = model._prob_history[:, prefix_len-self.args.gamma-1:prefix_len, :self.vocab_size].to(torch.float32)
                prob[:, 0, 0] = -1
                prob[:, 0, 1:self.args.gamma*2] = x[:, prefix_len-self.args.gamma+1:prefix_len+self.args.gamma]
                self.draft_forward_times += self.args.gamma
            else:
                x = model.generate(input_ids, 1)
                # print(f'Generating the target model outputs')
                prob = model._prob_history[:, prefix_len-self.args.gamma-1:prefix_len, :self.vocab_size].to(torch.float32)
                # ! the prob of the target model should be moved to a different device of the draft device to avoid deadlock
                prob = prob.to("cuda:1")
                self.target_forward_times += 1
            
            self.accelerator.wait_for_everyone()

            # verification
            all_prob = self.accelerator.gather(prob).to(device)
            # print(f'all_prob shape: {all_prob.shape}')
            draft_ids = all_prob[0, [0], 1:self.args.gamma*2].int()
            draft_prob = all_prob[[0], 1:, :]
            target_prob = all_prob[[1], 1:, :]

            if cur_mode:
                first_token = draft_ids[:, -self.args.gamma]
                torch.manual_seed(self.seed + prefix_len)

                r = torch.rand(1, device=device)
                if  r > target_prob[:, -1, first_token] / draft_prob[:, -1, first_token]:
                    # reject the first token
                    # print(f'Rejecting the first token')
                    t = sample(max_fn(target_prob[:, -1, :] - draft_prob[:, -1, :]))
                    prefix = torch.cat((input_ids, t), dim=1)
                    
                    # record the number of accepted tokens
                    self.num_acc_tokens.append(num_acc_token)
                    num_acc_token = 0
                    
                    if self.accelerator.is_main_process:
                        # rollback the small model kv cache
                        model.rollback(prefix_len)
                else:
                    # accept the first token, change the mode
                    # print(f'Accepting the first token and changing the mode')
                    cur_mode = False
                    prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma:]), dim=1)
                    num_acc_token += 1

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
                    # print(f'Accepting all guess tokens')
                    prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma:]), dim=1)
                    num_acc_token += self.args.gamma
                else:
                    # reject someone, change the mode
                    assert n < self.args.gamma
                    cur_mode = True
                    t = sample(max_fn(target_prob[:, n, :] - draft_prob[:, n, :]))
                    
                    prefix = torch.cat((input_ids[:, :prefix_len-self.args.gamma + n + 1], t), dim=1)
                    self.num_acc_tokens.append(num_acc_token + n)
                    num_acc_token = 0
                    # rollback both the large model and the small model kv cache
                    model.rollback(prefix_len - self.args.gamma +n+1)
            
            self.accelerator.wait_for_everyone()
            # print(f'Updated prefix shape: {prefix.shape}')
            
        return prefix

    
    @abstractmethod
    def eval(self):
        pass

    def color_print(self, content: str, color_number: int=4):
        # """print content with color. Some color numbers are listed: Gray: 0, Red: 1, Green: 2, Yellow: 3, Blue: 4."""
        if self.accelerator.is_main_process:
            print(f"\033[9{color_number}m{content}\033[0m")
