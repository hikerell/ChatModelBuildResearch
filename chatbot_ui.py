import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["ACCELERATE_USE_MPS_DEVICE"] = "True"

import sys
import argparse
import torch
import transformers
from transformers import pipeline, set_seed
from transformers import AutoTokenizer, OPTForCausalLM, GenerationConfig, AutoConfig
import gradio as gr


device_name = "cuda" if torch.cuda.is_available() else "cpu"
try:
    if torch.backends.mps.is_available():
        device_name = "mps"
except:
    pass
device = torch.device(device_name)

# tokenizer = transformers.AutoTokenizer.from_pretrained('/Users/hikerell/Workspace/aitech/deepspeed-train-models/shine-RLHF-20230414-on-opt-1.3b')
# model = transformers.AutoModelForCausalLM.from_pretrained("/Users/hikerell/Workspace/aitech/deepspeed-train-models/shine-RLHF-20230414-on-opt-1.3b",
#     # load_in_8bit=True,
#     # load_in_8bit=True,
#     # torch_dtype=torch.float16,
#     # torch_dtype=torch.float16,
#     # load_in_8bit_fp32_cpu_offload=True,
#     # device_map="auto",
#     # device_map="sequential",
#     # device_map='disk'
# )

# model = PeftModel.from_pretrained(
#     model, 
#     "tloen/alpaca-lora-7b", 
#     torch_dtype=torch.float16
# )

# print(f'set model to device {device} ...')
# model = model.to(device)
# print(f'set model to device {device} ... success!')


def generate_prompt(human: str, system: str=''):
    if system:
        return f"System: {system} \nHuman: {human} \nAssistant:"
    return f"Human: {human} \nAssistant:"

def get_generator(path):
    # tokenizer = AutoTokenizer.from_pretrained(path, fast_tokenizer=True)
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(path)
    model = OPTForCausalLM.from_pretrained(path,
                                           from_tf=bool(".ckpt" in path),
                                           # config=model_config).half()
                                           # torch_dtype=torch.float16,
                                           config=model_config)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    generator = pipeline("text-generation",
                         model=model,
                         tokenizer=tokenizer,
                         # device="cuda:0")
                         device=device)
    return generator

def get_model_response(generator, prompt, max_new_tokens):
    response = generator(prompt, max_new_tokens=max_new_tokens)
    return response

def process_response(response):
    output = str(response[0]["generated_text"])
    output = output.replace("<|endoftext|></s>", "")
    # all_positions = [m.start() for m in re.finditer("Human: ", output)]
    # place_of_second_q = -1
    # if len(all_positions) > num_rounds:
    #     place_of_second_q = all_positions[num_rounds]
    # if place_of_second_q != -1:
    #     output = output[0:place_of_second_q]
    return output
    
class Chatbot(object):
    def __init__(self) -> None:
        self.history = []
        self.generator = None

    def load(self, path):
        self.generator = get_generator(path)

    def evaluate(self, human, system='', max_tokens=256):
        print(f"max_tokens={max_tokens}")
        system = system.strip() if system else ''
        prompt = generate_prompt(human, system=system.strip())
        print(prompt)
        response = get_model_response(self.generator, prompt, max_new_tokens=max_tokens)
        output = process_response(response)
        print(output)
        return output
    
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        type=str,
                        required=True,
                        help="Directory containing model")
    args = parser.parse_args()

    chatbot = Chatbot()
    chatbot.load(args.path)

    gr.Interface(
        fn=chatbot.evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2, label="human", placeholder="hi."
            ),
            gr.components.Textbox(
                lines=2, label="system", placeholder=""
            ),
            gr.components.Slider(minimum=0, maximum=2048, step=8, value=256, label="max token length"),
            # gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
            # gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
            # gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k"),
            # gr.components.Slider(minimum=0, maximum=4, step=1, value=4, label="Beams"),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="chatbot",
        description=f"chatbot: {args.path}",
    ).launch(share=False)


if __name__ == "__main__":
    main()


