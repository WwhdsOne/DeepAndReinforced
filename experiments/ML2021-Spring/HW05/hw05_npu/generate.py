import torch


@torch.no_grad()
def greedy_decode(model, src_tokens, src_lengths, max_len, bos_idx, eos_idx):
    del src_lengths
    model.eval()
    encoder_outputs, src_key_padding_mask = model.encode(src_tokens)
    batch_size = src_tokens.size(0)
    generated = torch.full(
        (batch_size, 1),
        bos_idx,
        dtype=torch.long,
        device=src_tokens.device,
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=src_tokens.device)

    for _ in range(max_len):
        logits = model.decode(generated, encoder_outputs, src_key_padding_mask)
        next_tokens = logits[:, -1, :].argmax(dim=-1)
        next_tokens = torch.where(
            finished,
            torch.full_like(next_tokens, eos_idx),
            next_tokens,
        )
        generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
        finished |= next_tokens.eq(eos_idx)
        if bool(finished.all()):
            break

    hypotheses = []
    for row in generated[:, 1:]:
        eos_positions = row.eq(eos_idx).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            row = row[: eos_positions[0]]
        hypotheses.append(row.detach().cpu())
    return hypotheses
