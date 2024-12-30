from transformers import AutoConfig
import torch 


def unit_test_sigmoid_smoe(inp, gate_values, experts, output):
    out_test = torch.zeros_like(inp)

    for b in range(inp.shape[0]):
        for t in range(inp.shape[1]):
            for e in range(len(experts)):
                out_test[b][t] += gate_values[b][t][e]*experts[e](inp[b][t])
                
    if torch.allclose(output, out_test, atol=0.1) == False:
        with open("/cm/archive/namnv78/CUMO/test.txt", 'a') as f:
            f.write(f"\noutput\n{output}\nout_test\n{out_test}")
        print("Warning: Ouput of SMOE not correct!")
        return False
    return True

def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if 'llava' in config and 'llava' not in cfg.model_type:
        assert cfg.model_type == 'llama'
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)
