---
title: 'An Introduction to the Transformer Architecture'
date: 2025-06-05
permalink: /posts/2025/06/attention-is-all-you-need-explained/
tags:
  - machine learning
  - NLP
  - transformers
---
# Attention Is All You Need — A Proper Explanation

**By Atharva Yeole | June 2026

---

## How RNNs work

At every time step, the RNN takes two things:

- the current input x(t) suppose a word
- the hidden state h(t-1) (a vector that summarizes everything seen so far)

And it produces:

- a new hidden state h(t)
- hₜ = tanh(W_h · h(t-1) + W_x · x(t) + b)

NOTE: here (t) means at that given time not a function of time.

## What was actually broken with RNNs

The first one is the **vanishing gradient problem**. When you backpropagate through 50 time steps, your gradient gets multiplied by small numbers at each step. By the time it reaches step 1, it is basically zero. The model simply cannot learn long-range dependencies. It forgets.

The second problem is **RNNs are sequential by design**. Step 2 cannot start until step 1 finishes. You cannot parallelize this. Training on long sequences is genuinely slow, and in 2017 this was a real bottleneck.

LSTMs helped with the forgetting problem by adding a cell state and gates. But they were still sequential. Still slow. Still hitting limits on very long sequences.

This is the gap "Attention Is All You Need" (Vaswani et al., 2017) fills. The paper proposes dropping recurrence entirely and replacing it with a mechanism called **self-attention** that lets every token directly talk to every other token in one shot, with full parallelism.

## Q, K, V intuition

Imagine you are in a group of 4 friends and someone asks a question. You do not answer immediately. First you scan the room — you look at each person and internally decide how relevant they are to what you are about to say. Your old friend who knows the full context gets 70% of your attention. The new person who just arrived gets 5%. Then your answer is basically a weighted blend of what everyone in the room is contributing.
That scan is Q matching against K. The actual words each person would contribute to your answer is V. Your final response is the weighted mix.

**Q (Query)** -> What am I looking for?
**K (Key)** -> What do I advertise myself as?
**V (Value)** -> What do I actually hand over?

### Where do Q, K, V actually come from?

This tripped me up early. Q, K, and V are not three different inputs. They all come from the **same input** just that they are projected differently using three learned weight matrices.

X = token embeddings   (shape: seq_len × d_model)

Q = X · W_Q
K = X · W_K
V = X · W_V

Every token gets three versions of itself: a query version, a key version, and a value version. The network learns during training what these projections should look like. You never manually define what "a good key" is we use optimization function like gradient descent to figures it out.

## Scaled dot-product attention

Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V

**Step 1 — Compute QKᵀ (raw attention scores)**

Multiply Q and K (transposed). For a 3-token sentence this gives a 3×3 matrix where cell [i][j] is a score for how much token i should attend to token j. It is just a dot product — high dot product means the query and key point in similar directions in vector space, meaning they are relevant to each other.

**Step 2 — Divide by √d_k (the scaling step)**

When d_k (the dimension of each key vector) is large, say 512, dot products naturally grow very large in magnitude. Large inputs to softmax push the output toward one-hot distributions, basically 100% attention on one token and zero everywhere else. That kills gradient flow. Dividing by √d_k keeps values in a stable range so softmax produces smooth, distributed attention weights. Think of it as a temperature control knob.

**Step 3 — Softmax (convert scores to probabilities)**

Each row of the score matrix gets passed through softmax so values sum to 1. Now every token has a proper probability distribution over all other tokens.

**Step 4 — Multiply by V (gather information)**

Finally, multiply the attention weights by V. Each token's output becomes a weighted average of all Value vectors across the sequence.

### The attention score matrix

![Attention Matrix](/assets/AttentionMatrix.jpg)

## Multi-head attention(why one head is not enough)

A single attention head gives you one way of looking at the relationships in your sentence. But a sentence has multiple types of relationships happening simultaneously.

Take this sentence: *"The animal didn't cross the street because it was too tired."*

What does **"it"** refer to? The animal. That is a coreference relationship. Who is the subject of "cross"? The animal. That is syntactic. Why didn't it cross? Because of "tired." That is causal. One attention head can really only learn to model one type of relationship at a time.

Multi-head attention runs `h` separate attention heads in parallel, each with its own W_Q, W_K, W_V matrices. Each head gets a smaller dimensional slice to work with so the total compute stays the same. Then all heads are concatenated and projected back:

head_i       = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

MultiHead    = Concat(head_1, ..., head_h) · W_O

Nobody tells each head what to learn. They figure it out through training. When researchers visualize attention heads in trained Transformers, they consistently find heads that have specialized some track syntax, some track coreference, some look at neighboring tokens. It emerges from gradient descent alone.

## What one Transformer layer actually looks like

![One Transformer layer](/assets/TransformerLayer.jpeg)

*Based on Vaswani et al., "Attention Is All You Need," NeurIPS 2017.*
