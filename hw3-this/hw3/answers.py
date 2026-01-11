r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0,
        lr_sched_factor=0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 64
    hypers["seq_len"] = 128
    hypers["h_dim"] = 128
    hypers["n_layers"] = 3
    hypers["dropout"] = 0.4
    hypers["learn_rate"] = 3e-4
    hypers["lr_sched_factor"] = 0.5
    hypers["lr_sched_patience"] = 2
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    pass
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

We split the corpus into sequences instead of training on the whole text because:

1) Memory limits – training on one enormous sequence would require storing activations and gradients for all timesteps for backprop.

2) Backprop is very hard – long sequences make training slow and unstable due to vanishing/exploding gradients:

In an RNN, there is a hidden state $h_t$ that is updated at every time step according to

$$ h_t = f(h_{t-1}, x_t). $$

When backpropagating, the gradient w.r.t prevoius hidden state involves a product of many derivatives across time:

$$ \frac{\partial L}{\partial h_{t-k}} =
\frac{\partial L}{\partial h_t}
\prod_{i=t-k+1}^{t}
\frac{\partial h_i}{\partial h_{i-1}} $$

Multiplying many small values can make the product shrink and the model struggles to learn, while multiplying many large values can lead to extremely large gradients and unstable parameter updates.


3) More training samples – a long text yields many shorter input and target pairs, helping to optimize the model and get better generalization.

"""

part1_q2 = r"""
**Your answer:**

It is possible that the generated text shows memory longer than the sequence length because $S$ limits how far gradients are backpropagated, and not how long information can persist during the forward computation.

The hidden state can be carried forward across consecutive batches, such that the initial hidden state $h_0^{(i)}$ of batch $i$ is the final hidden state $h_t^{(i-1)}$ of the previous batch. During generation, we sample one character at a time, feed it back into the model, and propagate $h_t$ continuously.

Therefore, information from much earlier batches can influence later predictions through the evolving chain of hidden states, even when the training used truncated sequences.
"""

part1_q3 = r"""
**Your answer:**

We are not shuffling the order of batches when training because the training setup assumes that the tokens in the next batch are the direct continuation of the tokens in the previous batch. This allows the model to keep using the hidden state accumulated from the previous batch.

If we shuffle batches, the next batch will no longer continue the previous text, so the hidden state carried over from the previous batch will not match the new tokens. This makes training inconsistent and noisy, and it hurts the model’s ability to learn longer range dependencies.
"""

part1_q4 = r"""
**Your answer:**
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    # hypers["batch_size"] = 64
    # hypers["h_dim"] = 1024
    # hypers["z_dim"] = 512
    # hypers["x_sigma2"] = 2e-3
    # hypers["learn_rate"] = 2e-4
    # hypers["betas"] = (0.5,0.999)
    hypers["batch_size"] = 8
    hypers["h_dim"] = 512
    hypers["z_dim"] = 16
    hypers["x_sigma2"] = 1.0
    hypers["learn_rate"] = 2e-4
    hypers["betas"] = (0.5,0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
"""

part2_q2 = r"""
**Your answer:**
"""

part2_q3 = r"""
**Your answer:**
"""

part2_q4 = r"""
**Your answer:**
"""


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    pass
    # ========================
    return hypers


part3_q1 = r"""
Stacking encoder layers that use the sliding-window attention results in a broader context in the final layer beacuse it effectively implements information propagation.

While a single layer only captures local relationships (immediate neighbors within the window), each subsequent layer aggregates representations that already contain the information of their window neighbors. This allows distant information to travel step-by-step through the network. Just like in CNNs, stacking layers effectively increases the receptive field, allowing a token in the final layer to incorporate information from a wide span of the original input, despite only explicitly attending to a small local window at each step - the receptive field at the final layer grows linearly with the depth of the network. 
"""

part3_q2 = r"""
**Your answer:**
"""

# ==============
