import torch
import moe_launcher


def main():
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    tokens_num = 16
    hidden = 2048
    experts = 128
    inter = 768

    tokens = torch.randn(tokens_num, hidden, device=device, dtype=torch.float32)
    gate_w = torch.randn(hidden, experts, device=device, dtype=torch.float32)
    ffn1_w = torch.randn(experts, hidden, inter, device=device, dtype=torch.float32)
    ffn2_w = torch.randn(experts, inter, hidden, device=device, dtype=torch.float32)
    output = torch.empty(tokens_num, hidden, device=device, dtype=torch.float32)

    launcher = moe_launcher.create_launcher(tokens_num)
    launcher.create(tokens_num)
    err = launcher.launch(tokens, gate_w, ffn1_w, ffn2_w, output, tokens_num)
    if err != 0:
        raise RuntimeError(f"Launch failed, hipError={err}")

    torch.cuda.synchronize(device)
    print("Output[0, :8]:", output[0, :8].tolist())

    launcher.destroy()


if __name__ == "__main__":
    main()
