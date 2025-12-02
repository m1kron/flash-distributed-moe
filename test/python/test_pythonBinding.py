import torch
import flashMoeLauncher


def test_pythonBinding():
    torch.manual_seed(123)
    device = torch.device("cuda")

    tokens_num = 16
    hidden = 2048
    experts = 128
    inter = 768

    # Allocate tensors on same device (float32, contiguous)
    tokens = torch.randn(tokens_num, hidden, device=device, dtype=torch.float32)
    # Gate weights expected as [experts, hidden] in most implementations
    gate_w = torch.randn(experts, hidden, device=device, dtype=torch.float32)
    # Expert weights (placeholders; match your launcher’s expected layout)
    ffn1_w = torch.randn(experts, inter * 2, hidden, device=device, dtype=torch.float32)
    ffn2_w = torch.randn(experts, hidden, inter, device=device, dtype=torch.float32)
    output = torch.empty(tokens_num, hidden, device=device, dtype=torch.float32)

    launcher = flashMoeLauncher.MoeKernelLauncher()

    assert launcher.valid == False

    launcher.create(gate_w, ffn1_w, ffn2_w, tokens_num)

    assert launcher.valid == True

    # Call launch; some bindings return hipError code, others None
    launcher.launch(tokens, output)
    torch.cuda.synchronize(device)

    assert output.shape == (tokens_num, hidden)
    assert output.dtype == torch.float32

    launcher.destroy()

    assert launcher.valid == False
