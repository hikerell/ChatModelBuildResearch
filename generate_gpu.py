import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["ACCELERATE_USE_MPS_DEVICE"] = "True"

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


device_name = "cuda" if torch.cuda.is_available() else "cpu"
try:
    if torch.backends.mps.is_available():
        device_name = "mps"
except:
    pass
device = torch.device(device_name)

tokenizer = transformers.AutoTokenizer.from_pretrained('/Users/hikerell/Workspace/aitech/deepspeed-train-models/shine-RLHF-20230414-on-opt-1.3b')
model = transformers.AutoModelForCausalLM.from_pretrained("/Users/hikerell/Workspace/aitech/deepspeed-train-models/shine-RLHF-20230414-on-opt-1.3b",
    # load_in_8bit=True,
    # load_in_8bit=True,
    # torch_dtype=torch.float16,
    # torch_dtype=torch.float16,
    # load_in_8bit_fp32_cpu_offload=True,
    # device_map="auto",
    # device_map="sequential",
    # device_map='disk'
)

# model = PeftModel.from_pretrained(
#     model, 
#     "tloen/alpaca-lora-7b", 
#     torch_dtype=torch.float16
# )

print(f'set model to device {device} ...')
model = model.to(device)
print(f'set model to device {device} ... success!')


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


model.eval()


def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        **kwargs,
):
    prompt = f"Human: {instruction}\nAssistant: "
    # prompt = generate_prompt(instruction, input)
    print("---- [prompt start] ----")
    print(prompt)
    print("---- [prompt end] ----")
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(device)
    # print(inputs)
    # input_ids = inputs["input_ids"].cuda()
    input_ids = inputs["input_ids"].to(device)
    # generation_config = GenerationConfig(
    #     temperature=temperature,
    #     top_p=top_p,
    #     top_k=top_k,
    #     num_beams=num_beams,
    #     **kwargs,
    # )
    # with torch.no_grad():
    #     outputs = model.generate(
    #         **inputs,
    #         num_beams=5,
    #         max_new_tokens=50, 
    #         early_stopping=True,
    #         no_repeat_ngram_size=2)
    # # s = generation_output.sequences[0]
    # text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # # output = tokenizer.decode(s)
    # # return output.split("### Response:")[1].strip()
    # print(text_output[0])

    # inputs = tokenizer(prompt, return_tensors="pt")
    # outputs = model.generate(input_ids=input_ids, num_beams=5, 
    #                     max_new_tokens=50, early_stopping=True,
    #                     no_repeat_ngram_size=2)

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=50,
        )
    # print(f"outputs={outputs}")
    s = outputs.sequences[0]
    text_output = tokenizer.decode(s, skip_special_tokens=True)
    print(f"text_output={text_output}")
    return text_output


# gr.Interface(
#     fn=evaluate,
#     inputs=[
#         gr.components.Textbox(
#             lines=2, label="Instruction", placeholder="Tell me about alpacas."
#         ),
#         gr.components.Textbox(
#             lines=2, label="Input", placeholder="none"
#         ),
#         gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
#         gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
#         gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k"),
#         gr.components.Slider(minimum=0, maximum=4, step=1, value=4, label="Beams"),
#     ],
#     outputs=[
#         gr.inputs.Textbox(
#             lines=5,
#             label="Output",
#         )
#     ],
#     title="🦙🌲 Alpaca-LoRA",
#     description="Alpaca-LoRA is a 7B-parameter LLaMA model finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and makes use of the Huggingface LLaMA implementation. For more information, please visit [the project's website](https://github.com/tloen/alpaca-lora).",
# ).launch(share=False)

# Old testing code follows.


if __name__ == "__main__":
    # testing code for readme
    # for instruction in [
    #     "Tell me about alpacas.",
    #     "Tell me about the president of Mexico in 2019.",
    #     "Tell me about the king of France in 2019.",
    #     "List all Canadian provinces in alphabetical order.",
    #     "Write a Python program that prints the first 10 Fibonacci numbers.",
    #     "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
    #     "Tell me five words that rhyme with 'shock'.",
    #     "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
    #     "Count up from 1 to 500.",
    # ]:
    #     print("==== [start] ====")
    #     print("Instruction:", instruction)
    #     print("Response:", evaluate(instruction))
    #     print("==== [end] ====")
    #     print()

    while True:
        s = input("user: ").strip()
        if s in ['exit', 'q', 'quit']:
            break
        response = evaluate(s)
        print("Response:", response)
        print('---- + ----')

