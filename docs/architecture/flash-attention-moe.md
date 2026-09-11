# FlashAttention 与 Top-2 MoE 工作负载

仓库提供两个由真实 PyTorch 数学计算经 `torch.export → Torch-XLA → StableHLO → GC →
FC → target lowering` 编译的独立算子。二者不是手写 TISA 图，GC 只增加非 opaque semantic
region，内部 Matmul、reduce、elementwise、slice 和 transfer 仍作为 scheduler-visible TISA。

## FlashAttention

入口为 `examples.configurable_models.FlashAttention`，配置为
`configs/workloads/flash-attention.json`。输入采用 `[batch, heads, sequence, head_dim]` 的
Q/K/V 和可 broadcast 的 additive mask。

每个 KV block 立即更新 online softmax 状态，不保存完整概率矩阵：

```text
S_j = Q K_j^T / sqrt(d) + mask_j
m_new = max(m_old, rowmax(S_j))
P_j = exp(S_j - m_new)
l_new = l_old * exp(m_old - m_new) + rowsum(P_j)
O_new = O_old * exp(m_old - m_new) + P_j V_j
output = O / l
```

The shipped configuration uses Q/K/V=[1,2,8,16], query_block_size=4 and kv_block_size=8, producing 2 query blocks, 1 KV block and 2 score blocks. tile_size=8 splits head_dim into two tiles. Static slice, reshape and concatenate map output tiles to one or more source segments, so head_dim, query_length or key_length larger than tile_size does not require a full-tensor transform.


## Top-2 MoE

入口为 `examples.configurable_models.Top2MoE`，配置为 `configs/workloads/moe-top2.json`。
算子内部执行：

```text
logits = router(x)
probabilities = softmax(logits)
selected = deterministic_top2(logits)
weights = normalize(probabilities * selected)
expert_e = down_e(silu(gate_e(x)) * up_e(x))
output = sum_e(weights_e * expert_e)
```

caller 不再提供 routing mask；StableHLO `compare/select`、两个 row-max、归一化、四个 SwiGLU
expert 和 combine 均在图内。GC region 记录 `routing_contract=internal_router+normalized_top2`。

当前实现是数值精确的 top-2 sparse weighting，但为保持固定 shape StableHLO/TISA 合同，
四个 expert branch 都会执行，再由稀疏权重 combine。因此它适合验证 router、expert branch
依赖和调度，不应把周期解释为已有真实 token compaction 的硬件性能。动态 token gather/
scatter、expert capacity、overflow/drop policy 和按实际 token count 缩放 expert timing 仍是
后续独立合同。

## 运行

```bash
npu-ooo compile-and-sim \
  --config configs/workloads/flash-attention.json \
  --output-dir out/flash-attention

npu-ooo compile-and-sim \
  --config configs/workloads/moe-top2.json \
  --output-dir out/moe-top2
```

正式比较 static/dynamic 时，先 `compile` 一次，再让两个 `simulate` 复用相同 compile
package。`01_gc/canonical_graph.json` 的 `semantic_regions` 可核对算法、block/expert 数和
能力边界；最终调度粒度见 `03_tisa/tisa_program.json`。
