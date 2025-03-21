import os
import random
import argparse
import torch
import torch.nn.functional as F
import numpy as np
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

def seed_everything(seed: int):
    "set all random seed for reproducible results."
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def model_zoo(args):
    vocab_size = {
       
        "codegen-2b": 51200,
        "codegen-350m-multi": 51200,
        "codegen-350m-mono": 51200,
        "codegen-350-nl": 51200,
        
        "qwen-1.5b": 151936,
        "qwen-0.5b-coder":151936,
        "qwen-0.5b-instruct":151936,
        "qwen-0.5b":151936,

        "phi-1.5": 51200,
        "phi-2": 51200,
        "phi-1": 51200,

        "smo-1.7b": 49152,
        'smo-360m-instruct': 49152,
        'smo-360m': 49152,
    }
    
    zoo = {
        "codegen-2b": 'Salesforce/codegen-2B-multi',
        "codegen-350m-multi": 'Salesforce/codegen-350M-multi',
        "codegen-350m-mono": 'Salesforce/codegen-350M-mono',
        "codegen-350-nl":"Salesforce/codegen-350M-nl",

        "qwen-1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "qwen-0.5b-instruct": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "qwen-0.5b-coder":'Qwen/Qwen2.5-Coder-0.5B',
        "qwen-0.5b": "Qwen/Qwen2.5-0.5B",
        
        "phi-2":"microsoft/phi-2",
        "phi-1":"microsoft/phi-1",
        "phi-1.5":"microsoft/phi-1_5",

        "smo-1.7b":"HuggingFaceTB/SmolLM-1.7B-Instruct",
        'smo-360m-instruct':"HuggingFaceTB/SmolLM-360M-Instruct",
        'smo-360m':"HuggingFaceTB/SmolLM-360M",
    }

    args.draft_models =  [ zoo[draft_model] for draft_model in args.drafts]
    args.vocab_size = vocab_size[args.target]
    args.target_model = zoo[args.target]

def parse_arguments():
    """Specified arguments for running scripts."""
    parser = argparse.ArgumentParser(description='args for this file')
    
    parser.add_argument('--data_path', type=str, default="./../data/humaneval.jsonl", help='path to the dataset')

    parser.add_argument('--drafts',  type=str, nargs='+',  default=["codegen-350m-mono","codegen-350m-multi"], help='draft models for generating the first draft.')
    parser.add_argument('--target', type=str, default="codegen-2b", help='target model for generating the final code.')
    
    parser.add_argument('--exp_name', '-e', type=str, default="test", help='folder name for storing results.')
    parser.add_argument('--eval_mode', type=str, default="para_sd", help='eval mode.')
    parser.add_argument('--num_samples_per_task', '-n', type=int, default=2, help='num_samples for a task (prompt) in humaneval dataset.')
    parser.add_argument('--seed', '-s', type=int, default=1234, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--max_tokens', type=int, default=1024, help='max token number generated.')
    parser.add_argument('--temp', type=float, default=0.2, help='temperature for generating new tokens.')
    parser.add_argument('--top_k', type=int, default=0, help='top_k for ungreedy sampling strategy.')
    parser.add_argument('--top_p', type=float, default=0.95, help='top_p for ungreedy sampling strategy.')
    parser.add_argument('--gamma', type=int, default=4, help='guess time.')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for generating samples.')

    args = parser.parse_args()
    args.exp_name = os.path.join(os.getcwd(), "exp", args.exp_name)
    os.makedirs(args.exp_name, exist_ok=True)
    model_zoo(args)
    assert args.num_samples_per_task == len(args.draft_models), f"num_samples_per_task should be equal to the length of draft_models, but got {args.num_samples_per_task} and {len(args.draft_models)}"
    return args

def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits

def norm_logits(logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    if temperature == 0:
        idx = logits.argmax(dim=1)
        new_logits = torch.zeros_like(logits, device=logits.device)
        new_logits[:, idx] = 1
        return new_logits.float()
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs

def sample(probs : torch.Tensor, num_samples: int = 1):
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    return idx_next

def max_fn(x):
    """
        norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True) 
    return x_max / x_max_sum