# 19. MoE, routing, and expert parallelism

## MoE changes the compute path

MoE does not copy an FFN and average the copies. A router sends each token to a small number of experts. This increases total parameters without activating all of them, but adds load imbalance, token permutation, communication, and sharded checkpoint state.

~~~text
hidden state
    ↓
router logits → top-k expert ids + gate weights
    ↓
group by expert → expert FFN/MLP-prime → restore token order
    ↓
weighted sum + residual
~~~

Routing, expert computation, and order restoration form one contract. Output shapes alone cannot detect a token sent to the wrong expert.

## Controls and code

Fix expert count, top-k, shared experts, capacity, token-drop policy, router loss, and mesh before comparing variants. route_mlp, route_delta, and route_e2e change routing location; they should not also change width or training tokens.

ttt/models/ttt_e2e/moe.py implements routing and dispatch. grouped_gemm.py implements grouped matrix multiplication. ttt/distributed.py defines data/expert mesh axes and sharding. Start with no token drop for correctness, then measure expert counts, routing entropy, load ratio, and all-to-all time.

~~~bash
bash scripts/run_gpu_train.sh \
  --config configs/runs/mesh_tiny.json \
  --data outputs/sample-tokens.npy \
  --output outputs/mesh-tiny

bash scripts/validate_mesh.sh \
  outputs/mesh-validation.json \
  outputs/mesh-checkpoint
~~~

The report should include global and local expert shapes, routing counts, two training steps, restore consistency, communication time, and memory.

## Transition

The next chapter defines the experiment protocol that makes architecture and cost comparisons interpretable.
