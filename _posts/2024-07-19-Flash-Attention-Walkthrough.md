---
layout: post
title:  "Understanding Flash Attention"
date:   2024-07-19
categories: paper_walkthrough

---
# Table of Contents
1. [Introduction](#introduction)
2. [Background](#background)
    * [Standard Attention](#standard-attention)
    * [The GPU Architecture](#the-gpu-architecture)
3. [Flash Attention](#flash-attention)
    * [The Issue](#the-issue)
    * [Algorithm](#algorithm)
    * [Online Softmax](#online-softmax)
    * [Forward Pass](#forward-pass)
    * [Causal Masking](#causal-masking)
4. [Results](#results)
5. [Conclusion](#conclusion)


# Introduction
A consequence of the advancements of Deep Neural Networks (DNNs) in the past few years has been their increase in size. For example, GPT-1 was released by OpenAI in June 2018 and contained around 120 million parameters. In comparison, its successors GPT-2 (2019) and GPT-3 (2020) contain 1.5 billion and 175 billion parameters respectively.

This increase in size means that it is more expensive to store, train, and use these large models. From my own research, I can say that even training a tiny model (~4M parameters) takes over 2 days on 16 V100 GPUs. As a result, there have been many efforts in making DNNs more efficient through innovative approaches. One of such methods is [Flash Attention](https://arxiv.org/pdf/2205.14135) (introduced by Tri Dao et. al); of which exist three different versions at the time of writing. Today, we will walk through the [second version of Flash Attention](https://arxiv.org/pdf/2307.08691) and talk about how it manages to produce significant speedup on a model without affecting its accuracy.

*Hint : The trick is in being aware of the complexities of the underlying hardware*

> Disclaimer: This post will not cover the theoretical concepts regarding Attention and why its inclusion in Large Language Models (LLMs) is important. To learn more about that, I recommend reading [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). To motivate FA, however, we will review how attention is traditionally computed and investigate its inefficiencies. Furthermore, this post will **only cover the Forward Pass of FA**; the backward pass may be covered in a following post.


# Background
## Standard Attention
It is important to understand the standard way of computing attention (before FA). There are three matrices that are used in the computation of attention: Query (**Q**), Key (**K**), and Value (**V**) $$\in \mathbb{R}^{N\times d}$$ where $$N$$ is the sequence length of the model and $$d$$ is the head dimension. The equation for attention is:

$$
O = softmax(QK^{T}) V
$$

where softmax is defined row-wise as

$$
\sigma(z)_{i} = \frac{e^{z_{i}}}{\sum_{j=1}^K e^{z_{j}}}
$$

> Note:
> The exponential function grows quickly, often leading to overflow in the softmax function; however, a trick is used to avoid this issue based on the following equality:
>
> $$ \frac{e^{x+c}}{e^{x+c}+e^{y+c}} = \frac{e^{x}e^{c}}{e^{x}e^{c}+e^{y}e^{c}} = \frac{e^{x}e^{c}}{(e^{x}+e^{y})e^{c}} = \frac{e^{x}}{e^{x}+e^{y}} $$
>
> The softmax function is invariant to the choice of $$c$$. For convenience, the value of $$c$$ chosen in practice is the negation of the maximum value in the row (for row-wise softmax) which decreases the magnitude of all the values in the row, avoiding overflow.

The standard algorithm for computing attention is as follows:

---
**Input: $$Q, K, V$$**\
**Output: $$O$$**
1. Load $$Q$$ and $$K$$ from memory, compute $$S = QK^{T}$$ and store $$S$$ back into memory
2. Load $$S$$ from memory, compute $$P=softmax(S)$$ and store $$P$$ back into memory
3. Load $$P$$ and $$V$$ from memory, compute $$O=PV$$, and store $$O$$ to memory.
4. Return $$O$$

---
\
On the surface, this looks like a fine implementation. In an ideal world, where computations are done instantaneously, this algorithm would cause no problems. Unfortunately, we do not live in an ideal world and these algorithms are run on hardwares that demand their own tricks to perform at their best. In the next section, we will look deeper at the hardware on which these algorithms typically run, namely the GPU, and explore why this standard implementation is not efficient.


## The GPU Architecture
*To keep things simple, I will focus on NVIDIA GPUs throughout this guide; however, almost all concepts have an AMD equivalent. For example, warps can be replaced with wavefronts, CUDA can be replaced with HIP, Compute Units can be replaced with Execution Units, etc.* 

### Execution Model
GPUs are best-suited for working on embarassingly parallel workloads. Compared to a CPU, which might have anywhere between one to hundreds of threads working in parallel, a GPU could at any point have tens or hundreds of thousands of threads working in parallel.

Threads are launched in groups of 32 (64 for AMD) called *warps*. Each warp is issued a single instruction to execute; however, each thread within the warp will process different data, adhering to the Single-Instruction-Multiple-Data (SIMD) paradigm. If the number of elements in the data is not divisible by the number of threads launched, it is the responsibility of the programmer to mask the threads from performing any out-of-bounds operations.

Conceptually, threads can be launched in groups of up to 1024 (or 32 warps) called *Thread Block*s. These blocks are launched as part of a grid, which can be organized into a three-dimensional lattice of up to $$2^{31}-1$$ blocks in each dimension.

During runtime, each block gets scheduled to perform computations on a *Streaming Multiprocessor* (SM) which contains the required hardware to run many threads in parallel. Each SM has its own memory, memory controller, and many cores. The number of available SMs depends on the GPU used; for example, the A100 GPU has 108 SMs.

### Memory Hierarchy
Memory in the GPU is divided into multiple levels, constructing a neat hierarchy. Faster memories are placed closer to the chip, are more expensive, and offer much less storage than their slower counterparts. The memory types are typically comprised of the High-Bandwidth Memory (HBM), L2-Cache, L1-Cache, and Register File. Some of the levels in the hierarchy, like the register file, are not directly controlled by the developer through GPGPU compute libraries such as CUDA; therefore, we will simplify the memory model for our discussions and only refer to two separate memories. The first is High-Bandwidth Memory (HBM) and the second is a portion of L1-cache that can be reserved by the developer to be shared among threads in the same block, called Shared Memory (SMEM).

> Note:
> Persistence on the L2 cache, which resides between the HBM and SMEM on A100 GPUs is also [controllable by the developer](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#increased-l2-capacity-and-l2-residency-controls). However, this feature is not utilized by FA; therefore, we will not consider it for brevity.

The diagram below summarizes a simplified version of the GPU (A100) memory hierarchy:
<p align="center">
<img src="/assets/posts/flash_attention_walkthrough/GPUmemory.png" width="528" height="408">
</p>

As we move down the diagram, the storage capacity increases at the expense of lower bandwidth.

> Note:
> The GPU is not the only hardware architecture whose memory is designed in this hierarchical way. This design has a rich history and is used in almost every modern architecture today. Though it is out of the scope of this post, I recommend the avid reader to read about the *Memory Wall* to learn more about one of the biggest motivations behind this design.

The main idea is that the speed at which modern hardware can perform computation has long surpassed the speed at which data can be moved from and to memory. As a result, the pattern in the data movement, and not the computation speed of the chip, is the bottleneck for a large number of applications. Careful consideration is required when moving data to keep the hardware's compute cores fed and achieve maximum throughput.

The solution to this problem is to reduce the I/O to slower memory by keeping as much as the working set (data on which we are immediately operating) in the lower levels of memory (L1 and L2 cache) before writing the final result back to the highest (and slowest) level.

For more details about the GPU execution model or the memory hierarchy, I refer you to the [NVIDIA CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) or [AMD HIP](https://rocm.docs.amd.com/_/downloads/HIP/en/latest/pdf/) documentation.

# Flash Attention
With the required background out of the way, we can now discuss the issues of the Standard Attention algorithm and how Flash Attention fixes them.

## The Issue
Looking back at the algorithm for Standard Attention, the issue is clear: every time we interact with the memory (load/store), we are interacting with the HBM. In particular, the matrices $$S$$ and $$P$$ are stored into HBM which takes $$\mathcal{O}(N^{2})$$ time. This, coupled with the fact that the majority of the operations (everything other than the matrix multiply) are memory-bound, means that the standard attention algorithm is far from efficient given its naive use of the GPU memory hierarchy.

The goal of flash attention is to make use of the shared memory to fuse operations together such that the interactions with HBM are reduced, leading to faster end-to-end performance. That is, we would instead like to do something similar to:

---
**Input: $$Q$$, $$K$$, $$V$$**
**Output: $$O$$**
1. Load $$Q$$, $$K$$, and $$V$$
2. Compute $$S=QK^{T}$$ and store it into SMEM
3. Compute $$P=softmax(S)$$ and store it into SMEM
4. Compute $$O=PV$$ and store it back into HBM
5. Return $$O$$

---

Because of the limited size of shared memory, the full matrices will not fit in SMEM. As a result, the process is not as trivial as shown above and requires tiling; it is a method where we load small enough portions of the $$Q$$, $$K$$, and $$V$$ matrices to fit into SMEM and perform computations to achieve partial results before writing them back into HBM. A process of synchronization must take place at the end to update the partial results to the correct final result. We discuss the algorithm in more depth below.


## Algorithm
The FA algorithm is simple, it applies tiling by loading blocks of $$Q$$, $$K$$, and $$V$$ into SMEM, computing the corresponding attention, and updating the output without writing $$S$$ and $$P$$ back into HBM. There are, however, a few non-trivial problems that we must address.

### Online Softmax
Our first major problem is computing $$S=softmax(QK^{T})$$ through tiling. Splitting the work along the $$N$$ dimension (rows of $$Q$$) is embarrasingly parallel as there exists no dependency between the computations; however, splitting the work along the $$d$$ dimension is difficult since computing the row-wise softmax of $$QK^{T}$$ requires all the values along the row to be present. Consider the simple $$2\times8$$ matrix below representing some $$S$$.


<p align="center">
<img src="/assets/posts/flash_attention_walkthrough/Softmax_Row_Dependency.png" width="600" height="150">
</p>

Let us say that we want to organize the matrix into $$1\times2$$ tiles such that each tile is processed separately. The softmax of $$S$$ is then:

$$
softmax(S) = [s_{1} s_{2}]^{T}
$$

where $$s_{1}=[\frac{e^{a_{00}-m_{0}}}{\sum_{j=0}^{7}-e^{a_{0j}-m_{0}}}, ... ,\frac{e^{a_{07}-m_{0}}}{\sum_{j=0}^{7}-e^{a_{0j}-m_{0}}}]$$ is the softmax vector of the first row and $$s_{2}=[\frac{e^{a_{01}-m_{1}}}{\sum_{j=0}^{7}-e^{a_{1j}-m_{1}}}, ... ,\frac{e^{a_{07}-m_{1}}}{\sum_{j=0}^{7}-e^{a_{1j}-m_{1}}}]$$ is the softmax vector of the second row with $$m_{i}$$ being the maximum value in row $$i$$.

I hope it is clear that the value of the softmax for the different rows in $$S$$ are independent and can be parallelized; but both $$m$$ and and the denominator of the softmax operator make it necessary for tiles in the same row to synchronize for the correct output.

To compute the softmax using tiling, FA proposes an online softmax technique:

Assume we are working with some matrix $$S = [S_{1} S_{2}]$$. To fit our working set into SMEM, we must use tiling and compute partial results of $$S_{1}$$ and $$S_{2}$$. Each block in the tile will be of dimension $$B_{r}\times B_{c}$$, where $$B_{r}$$ is the number of rows in the block and $$B_{c}$$ is the number of columns in the block ($$S_{1}, S_{2} \in \mathcal{R}^{B_{r}\times B_{c}}$$). The online softmax algorithm is as follows:

---
$$m_{1} = rowmax(S_{1})$$\
$$l_{1} = rowsum(e^{S_{1}-m_{1}})$$\
$$O_{1} = e^{S_{1}-m_{1}}V_{1}$$

---
So far, nothing special has happened. We have found the maximum value within the first block (the first half of the rows), found the partial sum of the denominator, and stored the partial numerator of the softmax operator in the output.

Next, we compute the results of the second half of the rows, $$S_{2}$$. The main difference for this computation is that we already know the partial values for $$S_{1}$$, so we can compute the final result for $$S_{2}$$ and update the partial result of $$O_{1}$$ to the final correct result.

---
$$m = max(m_{1}, rowmax(S_{2}))$$\
$$l=e^{m_{1}-m}l_{1}+rowsum(e^{S_{2}-m})$$\
$$P_{2}=diag(l)^{-1}e^{S_{2}-m}$$\
$$O=diag(l)^{-1}(diag(e^{m_{1}-m})^{-1}O_{1}+e^{S_{2}-m}V_{2})$$

---
The first line above finds the maximum values of the entire rows. Then, the final denominator of the softmax function is computed by combining the sum of the rows in the second half of the matrix and the partial sum $$l_{1}$$ computed earlier. The value of the softmax for the second half of the matrix is computed in the third line. Finally, we compute the output in the last line. The first term is responsible for updating the partial results of the first half of the matrix ($$O_{1}$$), and the second term computes the results for the second half of the matrix.

### Forward Pass
With the issue of computing softmax through tiling out of the way, it is time we look at the algorithm of the forward pass as described in the paper. I will leave some comments along the way to make things clear.

---
**Input: $$Q$$, $$K$$, $$V$$, $$B_{r}$$, $$B_{c}$$**\
**Output: $$O$$, $$L$$**

Divide $$Q$$ into $$T_{r}=\lceil\frac{N}{B_{r}}\rceil$$ blocks $$Q_{1}...Q_{T_{r}}\in\mathcal{R}^{B_{r}\times d}$$\
Divide $$K$$ and $$V$$ into $$T_{c}=\lceil\frac{N}{B_{c}}\rceil$$ blocks $$K_{1}...K_{T_{c}}, V_{1}...V_{T_{c}} \in\mathcal{R}^{B_{c}\times d}$$\
Divide $$O\in\mathcal{R}^{N\times d}$$ and $$L$$ into $$T_{r}$$ blocks where $$O_{i}\in B_{r}\times d$$ and $$L_{i}\in \mathcal{R}^{B_{r}}$$
> Note: $$L$$ is the logsumexp variable that is used for the backward pass. We will cover its computation in the forward pass algorithm; however, we will not focus on it since this post does not cover the backward pass of FA.
>
> $$T_{r}$$ is the number of tiles we have going down the rows of $$Q$$, and $$T_{c}$$ is the number of tiles we have going down the columns of $$K$$ and $$V$$

---
So far, we have just divided the inputs and outputs into blocks which will be the unit of data we load into shared memory and operate on before writing the output back into HBM. The number of interactions with the HBM is also decided by the size of the matrices and the size of the blocks. Given a fixed matrix size, larger block sizes will result in fewer HBM interactions; therefore, there is an incentive to increase the block sizes as much as possible while still fitting all the data in SMEM. However, doing so will not always result in faster end-to-end times because of *Register Spilling*. We will not go into detail about this, but the algorithm can benefit from autotuning to pick the best block dimensions; this is left for future work by the FA authors. If implementing the algorithm in Triton, you can benefit from its autotuning facilities for this purpose. 

---
for $$1 \leq i \leq T_{r}$$ do\
&ensp;    Load $$Q^{(i)}$$ from HBM into SMEM\
&ensp;    Initialize $$O_{1}^{(i)} = (0)_{B_{r}\times d}, l_{0}^{(i)}=(0)_{B_{r}}, m_{1}^{(i)} = (-\infty)_{B_{r}}$$

> Note: $$Q^{(i)}$$ will live in shared memory while blocks of $$K$$ and $$V$$ are loaded for computation. $$O$$ and $$l$$ are initialized to zero because the values will be accumulated into them, while $$m$$ is set to $$-\infty$$ so any finite value will become the new maximum upon comparison with the first block

&ensp;   for $$1 \leq j \leq T_{c}$$ do\
&ensp;&ensp;    Load $$K_{j}$$ and $$V_{j}$$ from HBM into SMEM\
&ensp;&ensp;    Compute the $$S_{j}^{(i)}$$, $$m_{j}^{(i)}$$, and $$l_{j}^{(i)}$$, and finally $$O_{j}^{(i)}$$ in memory.\
> Note: The step above includes both computing the output for the current block and updating the block from the previous iteration as talked about in the *Online Softmax* section

&ensp;  endfor\
&ensp;  Compute $$O^{(i)}$$ and $$L^{(i)}$$ on chip\
&ensp;  Write $$O^{(i)}$$ to HBM as the $$i^{th}$$ block of $$O$$\
&ensp;  Write $$L^{(i)}$$ to HBM as the $$i^{th}$$ block of $$L$$\
endfor\
Return $$O$$ and logsumexp $$L$$

> Note: I have simplified the calculations happening on chip because we have discussed how to compute all the variables previously. For the full formulas, either refer to the previous steps in this walkthrough, or to the full algorithm in [the paper (Algorithm 1)](https://arxiv.org/pdf/2307.08691). I will also go into more details about the equations in a walkthrough of the Triton implementation in a later post

---

## Causal Masking
Before concluding, I want to cover another concept introduced in FA-V2 because it comes up in the Triton implementation.

Auto-regressive models, such as the models in the *GPT* family, need to apply a causal mask to the attention matrix $$S$$. This is because when these models compute attention for a given token, they only consider the tokens that precede it in the sequence. Therefore, all entries $$S_{ij}$$ where $$j>i$$ need to be ignored, and are set to $$-\infty$$.

Given FA already operates on blocks, we can skip the computation for every block where column indices are larger than the row indices. Furthermore, in blocks where the row indices are smaller than the column indices, we do not need to apply a mask since attention must be paid to all the elements in the rows of the block. Therefore, there exists only one block per row where a causal mask needs to be applied (assuming a square block); that is, the block where some of the row indices are smaller than some of the column indices.

According to the paper, this leads to around $$1.7-1.8\times$$ speedup compared to attention without the causal mask.

# Results
My main goal for this post has been to explain the algorithm and motivation behind FA; for the results and experiments, I refer you to the paper.

# Conclusion
Flash Attention's success and quick adoption is due to its simple algorithm that achieves significant speedup over standard attention implementations without any loss of accuracy (it is an exact attention algorithm, and not an approximation method). The key to the speedup in the algorithm is being IO-aware and taking advantage of the knowledge we possess of the underlying hardware on which the algorithm runs.

In the next post, we will dive deep into the code and explore the FA implementation in Triton.
