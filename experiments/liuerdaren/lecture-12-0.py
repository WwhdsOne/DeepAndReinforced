import torch

if __name__ == '__main__':
    batch_size = 1
    seq_len = 3
    input_size = 4
    hidden_size = 2
    num_layers = 1
    cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size,
                        num_layers=num_layers, batch_first=True)
    # (seq, batch, features)
    inputs = torch.randn(batch_size, seq_len, input_size)
    hidden = torch.zeros(num_layers, batch_size, hidden_size)

    out, hidden = cell(inputs, hidden)

    print('Output size:', out.shape)
    print('Output:', out)
    print('Hidden size:', hidden.shape)
    print('Hidden:', hidden)
