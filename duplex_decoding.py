import json, copy, time
import torch
import warnings, jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from datasets import load_dataset
from transformers.generation.utils import StoppingCriteriaList, LogitsProcessorList
from transformers.utils import (
    logging,
)

logger = logging.get_logger(__name__)

N_ahead=4
IDLE=0
WAIT_FOR_INPUT=1
GENERATION=2
PARREL=3

ENDFLAG="<SEP>"
STOPFLAG="</s>"

def generate_with_kvcache(prefix : torch.Tensor, 
                            gamma : int, 
                            model, past_key_values, prob_history, ) -> torch.Tensor:
    """ forward the model gamma times

    Args:
        prefix (torch.Tensor): the prefix
        gamma (int): how many times approx guesses

    Returns:
        Torch.Tensor: prefix+generated tokens
    """
    x = prefix

    for _ in range(gamma):
        q, past_key_values, prob_history = forward_with_kvcache(model, x, past_key_values, prob_history,)
        next_tok = q.argmax(dim=-1, keepdim=True)
        x = torch.cat((x, next_tok), dim=1)
    # from IPython import embed; embed()
    return x, past_key_values, prob_history

def forward_with_kvcache(model, input_ids, past_key_values, prob_history, ) -> torch.Tensor:
    if past_key_values is None:
        assert prob_history is None, f"{prob_history.shape}"
        # the first forward (prefill) returns the prompt's logits
        outputs = model(input_ids)
        prob_history = outputs.logits
        past_key_values = outputs.past_key_values
        last_q = prob_history[:, -1, :]
    else:
        # return the last token's logits
        cached_len = 0
        for kv in past_key_values:
            k, v = kv
            cached_len = k.shape[2]
            
        last_input_id = input_ids[:, cached_len:]
        if last_input_id.dim() == 1:
            last_input_id = torch.unsqueeze(last_input_id, 0)
        
        outputs = model(last_input_id, past_key_values=past_key_values, use_cache=True)
        
        not_cached_q = outputs.logits
        if not_cached_q.dim() == 2:
            not_cached_q = torch.unsqueeze(not_cached_q, 0) 
            
        prob_history = torch.cat([prob_history, not_cached_q], dim=1)
        
        last_q = not_cached_q[:, -1, :]
        past_key_values = outputs.past_key_values
    
    return last_q, past_key_values, prob_history

class DuplexModel():
    def __init__(self, model : torch.nn.Module, tokenizer, max_length: int = 4096,
                    do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, 
                    stopping_criteria=None, **kwargs) -> None:
        self._model = model
        self._model.eval()
        self.tokenizer = tokenizer
        self.device = self._model.device

        self._input_idx = None
        self._past_key_values = None
        self._prob_history = None

        self.asr_key_values = None
        self.asr_probs = None
        self.asr_idx = None

        self.status = IDLE
        self.idle_cnt = 0

        self.history = []
        self.history_all = []

        self.generate_flag = False
        self.print_len = 0
    @torch.no_grad()
    def reset(self):
        self._past_key_values = None
        self._prob_history = None

        self.asr_key_values = None
        self.asr_probs = None
        self.asr_idx = None

        self.is_output = False
        self.status = IDLE

    def reset_chat_history(self):
        save_file = "./data/duplex_logs/subject_normal_%s.jsonl"%(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time())))
        fw = jsonlines.open(save_file, "w")
        fw.write_all(self.history)
        fw.close()
        self.history = []
        self._input_idx = None
        self.print_len = 0
        self.is_length_limit = False
        self.logits_processor = LogitsProcessorList()

    @torch.no_grad()
    def generate(self, input : torch.Tensor, asr_channel: str, gamma : int) -> torch.Tensor:

        if self.status == IDLE and asr_channel is not None:
            if input is not None:
                new_input = torch.cat((input, self.tokenizer("<user>:"+asr_channel, return_tensors='pt').to('cuda')['input_ids'][:,1:]),dim=-1)
            else:
                new_input = self.tokenizer("<user>: " + asr_channel, return_tensors='pt').to('cuda')['input_ids']
            self.status = WAIT_FOR_INPUT
            # output, self._past_key_values, self._prob_history = generate_with_kvcache(new_input, gamma, self._model, self._past_key_values, self._prob_history,)
            output = new_input

        elif self.status == WAIT_FOR_INPUT:

            if asr_channel is not None:
                new_input = torch.cat((input, self.tokenizer(asr_channel, return_tensors='pt').to('cuda')['input_ids'][:,1:]),dim=-1)
                output, self._past_key_values, self._prob_history = generate_with_kvcache(new_input, gamma, self._model, self._past_key_values, self._prob_history,)
                self.idle_cnt = 0
            else:
                self.idle_cnt += 1
                output = input

            if self.idle_cnt > 1: #ENDFLAG in self.tokenizer.decode(output[0,-gamma:]):
                self.status = GENERATION
                new_input = torch.cat((output, self.tokenizer(" <assistant>:", return_tensors='pt').to('cuda')['input_ids'][:,1:]),dim=-1)
                self.print_len = len(self.tokenizer.decode(new_input[0]))
                output, self._past_key_values, self._prob_history = generate_with_kvcache(new_input, gamma, self._model, self._past_key_values, self._prob_history,)
            elif asr_channel is not None:
                self.rollback(output.shape[1]-gamma)
                output = output[:, :-gamma]

        elif self.status == GENERATION:
            
            if asr_channel != None:
                if self.asr_key_values == None:
                    new_input = torch.cat((input, self.tokenizer("<user>:"+asr_channel, return_tensors='pt').to('cuda')['input_ids'][:,1:]),dim=-1)
                else:
                    new_input = torch.cat((self.asr_idx, self.tokenizer(asr_channel, return_tensors='pt').to('cuda')['input_ids'][:,1:]),dim=-1)

                if self.asr_key_values == None and self._past_key_values != None:
                    self.asr_key_values = [(i[0], i[1]) for i in self._past_key_values]
                    self.asr_idx = input[:,:-1]
                    self.asr_probs = self._prob_history

            output, self._past_key_values, self._prob_history = generate_with_kvcache(input, gamma, self._model, self._past_key_values, self._prob_history,)

            if asr_channel != None:
                
                assert self.asr_key_values[0][0].shape[2] == self.asr_idx.shape[1]
                assert new_input.shape[1] != self.asr_idx.shape[1]

                self.asr_idx, self.asr_key_values, self.asr_probs = generate_with_kvcache(new_input, gamma, self._model, self.asr_key_values, self.asr_probs,)
                self.asr_idx = self.asr_idx[:, :-1]

                self.idle_cnt = 0
            else:
                self.idle_cnt += 1

            # ENDFLAG in self.tokenizer.decode(self.asr_idx[0,-gamma:]):
            if self.idle_cnt > 2 and self.asr_idx is not None:
                assert self.asr_key_values[0][0].shape[2] == self.asr_idx.shape[1]
                
                new_input = torch.cat((self.asr_idx, self.tokenizer("\n<assistant>:", return_tensors='pt').to('cuda')['input_ids'][:,1:]),dim=-1)
                self.print_len = len(self.tokenizer.decode(new_input[0]))
                new_input = torch.cat((self.asr_idx, self.tokenizer("\n", return_tensors='pt').to('cuda')['input_ids'][:,1:]),dim=-1)
                output, self._past_key_values, self._prob_history = generate_with_kvcache(new_input, gamma, self._model, self.asr_key_values, self.asr_probs,)

                self.asr_probs = None
                self.asr_key_values = None
                self.asr_idx = None

                self.status = GENERATION
            elif self.asr_idx is not None and asr_channel != None:
                self.asr_rollback(self.asr_idx.shape[1]-gamma)
                assert self.asr_key_values[0][0].shape[2] == self.asr_idx.shape[1]

        else:
            output = input

        self._input_idx = output
        return output
    
    @torch.inference_mode()
    def update_history(self):
        if len(self.history) > 0:
            if self.history[-1]["content"][-4:] != "</s>":
                self.history[-1]["content"] += "</s>"
            
            history_old = copy.deepcopy(self.history)
            self.history = []
            for i in range(0, len(history_old), 2):
                if history_old[i]["content"] == "<idle>":
                    if i + 1 < len(history_old) and history_old[i+1]["content"].strip(" .\n,") in ["<idle>", "<idle></s>", "</s>", "idle", "idle</s>"]:
                        self.generate_flag = False
                        continue
                    else:
                        self.history.append(history_old[i])
                        if i + 1 < len(history_old):
                            self.history.append(history_old[i+1])
                        self.generate_flag = True

                else:
                    self.history.append(history_old[i])
                    if i + 1 < len(history_old):
                        self.history.append(history_old[i+1])
                    self.generate_flag = True

    @torch.inference_mode()
    def stream_chat(self, tokenizer, query: str, max_length: int = 4096,
                    do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, 
                    stopping_criteria=None, **kwargs):

        self.new_input = None
        if self.generate_flag is True and query is not None:
            self.update_history()
            
            if self.generate_flag is False and query in ["<idle>"]:
                return 2
            elif query not in ["<idle>"]:
                self.generate_flag = True
            self.history.append({"role": "user", "content": query})
            self.history_all.append({"role": "user", "content": query, "timestamp": time.time()})
            
            self.new_input = query
            if self._input_idx is not None and self._input_idx.shape[-1] >= max_length:
                self.is_length_limit = True
                return 1

            self.history.append({"role": "assistant", "content": ""})
            self.history_all.append({"role": "assistant", "content": ""})

        elif self.generate_flag is False and query is not None and query not in ["<idle>"]:
            self.generate_flag = True
            self.history.append({"role": "user", "content": query})
            self.history_all.append({"role": "user", "content": query, "timestamp": time.time()})
           
            self.new_input = query
            if self._input_idx is not None and self._input_idx.shape[-1] >= max_length:
                self.is_length_limit = True
                return 1

            self.history.append({"role": "assistant", "content": ""})
            self.history_all.append({"role": "assistant", "content": ""})
        else:
            return 2
        if logits_processor is None:
            self.logits_processor = LogitsProcessorList()

        return 0
    
    @torch.inference_mode()
    def stream_generate(
            self,
    ):
        
        if self.generate_flag is False or self.is_length_limit is True:
            return None, None

        self.new_input = None if self.new_input == "<idle>" else self.new_input
        output = self.generate(self._input_idx, self.new_input, N_ahead)

        if self.status == IDLE or self.status == WAIT_FOR_INPUT:
            cut_len = 0 if output is None else len(self.tokenizer.decode(output[0]))
        else:
            cut_len = self.print_len

        response = self.tokenizer.decode(output[0])
        self.print_len = len(response)
        self.new_input = None
        if self.history_all[-1]["content"] == "":
            self.history_all[-1]["timestamp"] = time.time()
        self.history[-1]["content"] += response[cut_len:]
        self.history_all[-1]["content"] += response[cut_len:]

        print("------------------response---------------------")
        print(response)

        if STOPFLAG in self.tokenizer.decode(output[0,-N_ahead:]):
            self.status = IDLE

        return response[cut_len:], self.history
    
    @torch.no_grad()
    def rollback(self, end_pos : int):
        past_key_values_trimmed = []
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv
            # NOTE() the indexing is specific for bloom. This won't work for other models
            # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
            
            # k, v (batch, head, seq, hidden_dim)
            k = k[:, :, :end_pos, :]
            v = v[:, :, :end_pos, :]
            kv_trimmed = (k, v)
            past_key_values_trimmed.append(kv_trimmed)
        
        self._past_key_values = past_key_values_trimmed
        self._prob_history = self._prob_history[:, :end_pos, :]

    @torch.no_grad()
    def asr_rollback(self, end_pos : int):
        past_key_values_trimmed = []
        assert self.asr_key_values
        for kv in self.asr_key_values:
            k, v = kv
            # NOTE() the indexing is specific for bloom. This won't work for other models
            # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
            
            # k, v (batch, head, seq, hidden_dim)
            k = k[:, :, :end_pos, :]
            v = v[:, :, :end_pos, :]
            kv_trimmed = (k, v)
            past_key_values_trimmed.append(kv_trimmed)
        
        self.asr_key_values = past_key_values_trimmed
        self.asr_probs = self.asr_probs[:, :end_pos, :]
        self.asr_idx = self.asr_idx[:, :end_pos]

        assert self.asr_key_values[0][0].shape[2] == self.asr_idx.shape[1]

if __name__ == "__main__":

    model_path = "/mnt/afs/wxu/output/duplex/20240830114635/checkpoint-2000/" #"/mnt/afs/wxu/checkpoints/openbmb/MiniCPM-2B-sft-bf16/" #
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='cuda',trust_remote_code=True) # LlamaForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    cache_model = DuplexModel(model, tokenizer)
    
    with open("/mnt/afs/wxu/datasets/xinrongzhang2022/Duplex-UltraChat/regeneration_small.jsonl", "r") as file:
        dataset = file.readlines()

    for idx, line in enumerate(dataset):
        if idx == 0 or idx == 1:
            continue
        if idx > 10:
            break
    
        data = json.loads(line)
        cache_model.reset()

        input_idx = None
        for turn in data["output"]:
        
            if turn["role"] == "assistant":
                continue

            content = turn["content"]
            print("USER: ",content)
            if content == "<idle>":
                content = None

            input_idx = cache_model.generate(input_idx, content, N_ahead)

            if cache_model.status == GENERATION:
                print("                 Assistant: "+tokenizer.decode(input_idx[0][-N_ahead:]))


        while input_idx.shape[1] < 4096:

            input_idx = cache_model.generate(input_idx, content, N_ahead)
            print("                 Assistant: "+tokenizer.decode(input_idx[0][-N_ahead:]))

            if cache_model.status == IDLE:
                break

        print(tokenizer.decode(input_idx[0]))
