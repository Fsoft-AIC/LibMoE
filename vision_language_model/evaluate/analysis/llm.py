from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def generate_text(model_path, prompt, temperature=0.7, top_p=0.8, top_k=None, repetition_penalty=1.05, max_tokens=512):
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Set sampling parameters
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty, max_tokens=max_tokens)

    # Initialize the model
    llm = LLM(model=model_path, max_model_len=16368)

    # Prepare your prompt and messages
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate outputs
    outputs = llm.generate([text], sampling_params)

    # Return generated text
    for output in outputs:
        return output.outputs[0].text

# Example usage
model_path = "/cm/archive/anonymous/checkpoints/benchmarks/DeepSeek-R1-Distill-Qwen-32B"
prompt = "Tell me something about large language models."
generated_text = generate_text(model_path, prompt, top_k=50)
print(generated_text)
breakpoint()