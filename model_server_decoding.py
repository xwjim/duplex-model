import asyncio
import os
from duplex_decoding import DuplexModel
import torch
import websockets
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer

model_dir = "minicpm-duplex"
device = "cuda" # for linux server
#device = "mps" for mac book
# tokenizer = LlamaTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda")
model_path = "/mnt/afs/wxu/output/duplex/20240911010723/checkpoint-800" #"/mnt/afs/wxu/checkpoints/openbmb/MiniCPM-2B-sft-bf16/" #
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='cuda',trust_remote_code=True) # LlamaForCausalLM
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = DuplexModel(model, tokenizer)

max_length = 512

top_p = 0.8
temperature = 0.8
top_k = 0
int_out_token = 2


async def echo(websocket, path):
    is_ordinary = False
    async for message in websocket:
        message = message.strip()

        if not message:
            continue

        # new duplex session
        if message == "<!new_session!>":
            print("New session started.")
            model.reset_chat_history()
            is_ordinary = False
            continue

        # new ordinary session
        if message == "<!new_ordinary_session!>":
            print("New ordinary session started.")
            model.reset_chat_history()
            is_ordinary = True
            continue
        ret_code = model.stream_chat(tokenizer, message, max_length=max_length, top_p=top_p,
                          temperature=temperature, top_k=top_k)
                          
        if ret_code == 1:
            print("You have reached the max length limit. Please start a new chat!")
            try:
                await websocket.send("<!too_long!>")
            except:
                pass
        if ret_code == 0:
            server_resp = ""
            for i in range(max_length if is_ordinary else int_out_token):
                response, history = model.stream_generate()
                if "</s>" in response:
                    model.generate_flag = False
                    server_resp += response.split("</s>")[0]
                    break
                else:  
                    server_resp += response

            # server_resp = server_resp.replace(".", "\n")
            if server_resp and server_resp not in ['<idle>', "</s>", ' <idle>', ""]:
                try:
                    await websocket.send(server_resp)
                except:
                    pass


start_server = websockets.serve(echo, "localhost", 8765)
print("Server started.")

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()