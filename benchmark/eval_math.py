import os
import re
import sys
sys.path.append(os.path.join(sys.path[0], "../"))
import torch
import json
import tqdm
import time
import random
from src.util import seed_everything, parse_arguments
from src.engine import Decoding
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

class EvalMath500(Decoding):
    def __init__(self, args):
        super().__init__(args)
        
        # Set up prompting strategy
        self.ANSWER_TRIGGER = "Therefore, the answer is"
        self.prompt = self.create_demo_text(ANSWER_TRIGGER=self.ANSWER_TRIGGER)

        # load relative resources
        self.load_tokenizer()
        self.load_data()
        self.load_model()

    def create_demo_text(self, n_shot=3, ANSWER_TRIGGER="Therefore, the answer is"):
        """Create few-shot examples for the math500 dataset."""
        examples = [
            {
                "problem": "What is the value of $\\frac{1}{3} \\cdot \\frac{3}{4} \\cdot \\frac{4}{5}$?",
                "solution": "We have \n$\\frac{1}{3} \\cdot \\frac{3}{4} \\cdot \\frac{4}{5} = \\frac{1 \\cdot 3 \\cdot 4}{3 \\cdot 4 \\cdot 5} = \\frac{12}{60} = \\frac{1}{5}$",
                "answer": "\\frac{1}{5}"
            },
            {
                "problem": "Find the slope of the line passing through the points $(3, 5)$ and $(7, 9)$.",
                "solution": "The slope of a line passing through the points $(x_1, y_1)$ and $(x_2, y_2)$ is given by $m = \\frac{y_2 - y_1}{x_2 - x_1}$.\nSubstituting the given points, we get $m = \\frac{9 - 5}{7 - 3} = \\frac{4}{4} = 1$.",
                "answer": "1"
            },
            {
                "problem": "Compute the derivative of $f(x) = 3x^2 - 2x + 5$.",
                "solution": "Using the power rule of differentiation, we have:\n$f'(x) = 3 \\cdot 2x^{2-1} - 2 \\cdot 1x^{1-1} + 0$\n$f'(x) = 6x - 2$",
                "answer": "6x - 2"
            }
        ]
        
        demo_text = ""
        for i in range(min(n_shot, len(examples))):
            demo_text += (
                "Problem: " + examples[i]["problem"] + "\n"
                "Solution: " + examples[i]["solution"] + " "
                + ANSWER_TRIGGER + " "
                + examples[i]["answer"] + "\n\n"
            )
        
        return demo_text

    def clean_answer(self, answer_str):
        """Clean the answer string by removing unnecessary spaces and symbols."""
        # Fix: Replace backslash escapes properly in replacement strings
        answer_str = re.sub(r'\s*\\frac\s*', r'\\frac', answer_str)
        answer_str = re.sub(r'\s*\\pi\s*', r'\\pi', answer_str)
        answer_str = re.sub(r'\s*\\left\s*', r'\\left', answer_str)
        answer_str = re.sub(r'\s*\\right\s*', r'\\right', answer_str)
        
        # Remove trailing periods and unnecessary spaces
        answer_str = answer_str.strip().rstrip('.')
        
        return answer_str

    def extract_answer(self, completion):
        """Extract the answer from the model completion."""
        # Try to find answer after the trigger
        if self.ANSWER_TRIGGER.lower() in completion.lower():
            parts = completion.lower().split(self.ANSWER_TRIGGER.lower(), 1)
            answer = parts[1].strip()
        else:
            # If no trigger, look for boxed answers which are common in math
            boxed_match = re.search(r'\\boxed{(.*?)}', completion)
            if boxed_match:
                answer = boxed_match.group(1).strip()
            else:
                # Otherwise, take the last few sentences as the answer
                sentences = re.split(r'[.!?]', completion)
                answer = sentences[-2].strip() if len(sentences) > 1 else completion.strip()
        
        # Clean up the answer
        return self.clean_answer(answer)

    def load_data(self):
        """Load the math500 dataset."""
        self.color_print(f"Loading Math500 data...", 3)
        data = []
        try:
            with open(os.path.join(self.args.data_path)) as f:
                for line in f.readlines():
                    try:
                        datum = json.loads(line)
                        datum["input_text"] = self.preprocess(datum["problem"])
                        encode_special_token_flag = not ("Llama-3.1" in self.args.drafts and "Llama-3.1" in self.args.target)
                        input_ids = self.tokenizer.encode(datum["input_text"], add_special_tokens=encode_special_token_flag)
                        datum["input_ids"] = torch.tensor(input_ids).unsqueeze(0)
                        datum["ground_truth"] = self.clean_answer(datum["answer"])
                        data.append(datum)
                    except Exception as e:
                        self.color_print(f"Error processing datum: {e}", 1)
                        continue
        except Exception as e:
            self.color_print(f"Error loading data: {e}", 1)
        
        self.data = data[:15]
        self.color_print(f"Loaded {len(self.data)} examples", 2)

    def preprocess(self, problem_text):
        """Preprocess the problem text by adding the prompt and instruction."""
        text = self.prompt + "Problem: " + problem_text + "\n" + "Solution:"
        return text

    def postprocess(self, input_text, output_text):
        """Extract the answer from the model's output."""
        if self.tokenizer.bos_token is not None:
            generation = output_text[len(input_text)+len(self.tokenizer.bos_token)+1:] # tokenizer will add a '<s> ' at the beginning of the text.
        else:
            generation = output_text[len(input_text):]
        
        try:
            return self.extract_answer(generation)
        except Exception as e:
            self.color_print(f"Error extracting answer: {e}", 1)
            return "ERROR_EXTRACTING_ANSWER"
             
    @torch.no_grad()
    def eval(self):
        """Evaluate the model on the math500 dataset."""
        if self.args.eval_mode == "para_sd":
            decoding = self.parallel_speculative_decoding
        else:
            raise NotImplementedError
        
        out_path = os.path.join(self.args.exp_name, f"{self.args.eval_mode}_math500.jsonl")
        out_f = open(out_path, "a")
        wall_times = {"time":[], "num_tokens":[]}
        
        while self.seed in self.seed_set:
            self.seed = random.randint(0, 1000000)
        seed_everything(self.seed)
        self.seed_set.add(self.seed)
        
        acc = 0
        for idx, datum in tqdm.tqdm(enumerate(self.data), total=len(self.data), disable=not self.accelerator.is_main_process, ncols=50):
            input_ids = datum["input_ids"]
            torch.cuda.synchronize()
            start_time = time.time()
            generate_ids = decoding(input_ids)
            torch.cuda.synchronize()
            end_time = time.time()
            
            if self.accelerator.is_main_process:
                if idx != 0:
                    # skip the first prompt time consumption
                    wall_times["time"].append(end_time-start_time)
                    wall_times["num_tokens"].append(generate_ids.shape[1] - input_ids.shape[1])
                
                try:
                    predicted_answer = self.postprocess(datum["input_text"], self.tokenizer.decode(generate_ids[0, :]))
                    ground_truth = datum["ground_truth"]
                    
                    # Check if answers match (this is a simple exact match - might need more sophisticated comparison for math)
                    is_correct = (predicted_answer.lower() == ground_truth.lower())
                    if is_correct:
                        acc += 1
                    
                    out_f.write(json.dumps({
                        "problem": datum["problem"],
                        "time": end_time-start_time,
                        "new_tokens": generate_ids.shape[1] - input_ids.shape[1],
                        "ground_truth": ground_truth,
                        "predicted_answer": predicted_answer,
                        "correct": is_correct,
                        "subject": datum.get("subject", ""),
                        "level": datum.get("level", ""),
                        "unique_id": datum.get("unique_id", "")
                    }, ensure_ascii=False) + "\n")
                except Exception as e:
                    self.color_print(f"Error in evaluation: {e}", 1)
                    out_f.write(json.dumps({
                        "problem": datum.get("problem", ""),
                        "error": str(e)
                    }, ensure_ascii=False) + "\n")
            
            out_f.flush()
        
        if self.accelerator.is_main_process:
            self.color_print(f"Accuracy: {acc / len(self.data):.4f}", 2)
        
        out_f.close()
        
        self.color_print(f"current eval mode: {self.args.eval_mode}", 0)
        self.color_print(f"draft model forward times: {self.draft_forward_times}", 2)
        
        self.accelerator.wait_for_everyone()
        
        if (self.accelerator.num_processes == 1 and self.accelerator.is_main_process) or (self.accelerator.num_processes == 2 and not self.accelerator.is_main_process):
            print(f"\033[92mtarget model forward times: {self.target_forward_times}\033[0m")
        
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process and wall_times["time"]:
            speed = sum(wall_times["num_tokens"]) / sum(wall_times["time"])
            speed_std = (torch.tensor(wall_times["num_tokens"]) / torch.tensor(wall_times["time"])).std().item()
            self.color_print(f"generate speed (tokens / second): {speed:.2f} with std {speed_std}", 2)


if __name__ == "__main__":
    args = parse_arguments()
    alg = EvalMath500(args)
    alg.eval()