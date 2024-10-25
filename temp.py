from exo.inference.mlx.sharded_inference_engine import MLXDynamicShardInferenceEngine
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from exo.helpers import DEBUG
import os
import asyncio
import numpy as np


# An inference engine should work the same for any number of Shards, as long as the Shards are continuous.
async def test_inference_engine(inference_engine_1: InferenceEngine, inference_engine_2: InferenceEngine, model_id: str):
  prompt = "In a single word only, what is the last name of the current president of the USA?"
  resp_full, inference_state_full, _ = await inference_engine_1.infer_prompt("A", shard=Shard(model_id=model_id, start_layer=0, end_layer=31, n_layers=32), prompt=prompt)
  next_resp_full, _next_inference_state_full, _ = await inference_engine_1.infer_tensor(
    "A",
    shard=Shard(model_id=model_id, start_layer=0, end_layer=31, n_layers=32),
    input_data=resp_full,
    inference_state=inference_state_full,
  )

#   pp = 15
#   resp1, inference_state_1, _ = await inference_engine_1.infer_prompt("B", shard=Shard(model_id=model_id, start_layer=0, end_layer=pp, n_layers=32), prompt=prompt)
#   resp2, inference_state_2, _ = await inference_engine_2.infer_tensor(
#     "B",
#     shard=Shard(model_id=model_id, start_layer=pp + 1, end_layer=31, n_layers=32),
#     input_data=resp1,
#     inference_state=inference_state_1,
#   )
#   resp3, inference_state_3, _ = await inference_engine_1.infer_tensor(
#     "B",
#     shard=Shard(model_id=model_id, start_layer=0, end_layer=pp, n_layers=32),
#     input_data=resp2,
#     inference_state=inference_state_2,
#   )
#   resp4, _inference_state_4, _ = await inference_engine_2.infer_tensor(
#     "B",
#     shard=Shard(model_id=model_id, start_layer=pp + 1, end_layer=31, n_layers=32),
#     input_data=resp3,
#     inference_state=inference_state_3,
#   )

#   assert np.array_equal(resp_full, resp2)
#   assert np.array_equal(next_resp_full, resp4)


# asyncio.run(test_inference_engine(
#   MLXDynamicShardInferenceEngine(HFShardDownloader()),
#   MLXDynamicShardInferenceEngine(HFShardDownloader()),
#   "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
# ))

# import tinygrad
# import os
# from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
# tinygrad.helpers.DEBUG.value = int(os.getenv("TINYGRAD_DEBUG", default="0"))
# asyncio.run(
# test_inference_engine(
#     TinygradDynamicShardInferenceEngine(HFShardDownloader()),
#     TinygradDynamicShardInferenceEngine(HFShardDownloader()),
#     "TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R",
# )
# )


# from tinygrad import Tensor
# from tinygrad.dtype import dtypes
# import numpy as np
# import mlx.core as mx

# w = np.random.randint(0, 9, size=(1024, 512), dtype=np.uint32)
# s = np.random.rand(1024, 64).astype(np.float16)
# b = np.random.rand(1024, 64).astype(np.float16)
# x = np.random.rand(120, 4096).astype(np.float16)

# def quantized_matmul_tg(x, w_packed, scales, biases, width=4, groups=64):
#     """
#     Perform quantized matrix multiplication using tinygrad Tensors with shift operators.

#     Parameters:
#     - x: Tensor of shape (M, K), input activations.
#     - w_packed: Tensor of shape (N, K_packed), packed quantized weights (dtype=dtypes.int32).
#     - scales: Tensor of shape (N, K // groups), scales for dequantization (dtype=dtypes.float32).
#     - biases: Tensor of shape (N, K // groups), biases for dequantization (dtype=dtypes.float32).
#     - width: int, number of bits per quantized value (default is 4 bits).
#     - groups: int, number of quantization groups (default is 64).

#     Returns:
#     - output: Tensor of shape (M, N), result of the quantized matrix multiplication.
#     """
#     M, K = x.shape
#     N, K_packed = w_packed.shape

#     num_values_per_uint32 = 32 // width  # E.g., for width=4, this is 8
#     K_unpacked = K_packed * num_values_per_uint32
#     num_groups = K // groups
#     packs_per_group = groups // num_values_per_uint32  # Number of uint32 packs per group

#     assert K == K_unpacked, f"Mismatch in K dimensions: {K} vs {K_unpacked}"
#     assert scales.shape == (N, num_groups), f"Scales must have shape (N, {num_groups}), got {scales.shape}"
#     assert biases.shape == (N, num_groups), f"Biases must have shape (N, {num_groups}), got {biases.shape}"
#     assert K % groups == 0, "K must be divisible by the number of groups"

#     # Prepare bitmask
#     bitmask = (1 << width) - 1  # E.g., for width=4, bitmask=15

#     # Reshape x for group-wise processing
#     x_grouped = x.reshape(M, num_groups, groups)  # Shape: (M, num_groups, groups)

#     # Initialize the output matrix
#     output = Tensor.zeros((M, N), dtype=dtypes.float16)

#     # Prepare shift amounts
#     shift_list = [i * width for i in range(num_values_per_uint32)]

#     # Process each group
#     for g in range(num_groups):
#         # Extract scales and biases for the current group
#         scale_g = scales[:, g].reshape(N, 1)  # Shape: (N, 1)
#         bias_g = biases[:, g].reshape(N, 1)   # Shape: (N, 1)

#         # Extract the packed weights for the current group
#         pack_start = g * packs_per_group
#         pack_end = pack_start + packs_per_group
#         w_packed_group = w_packed[:, pack_start:pack_end]  # Shape: (N, packs_per_group)

#         # Initialize a list to collect unpacked values
#         unpacked_values = []

#         # Unpack the quantized weights
#         for shift_amount in shift_list:
#             # Perform the shift and mask operations
#             shifted = w_packed_group >> shift_amount  # Broadcasting scalar shift_amount
#             masked = (shifted & bitmask).cast(dtypes.float16)
#             masked = masked.reshape(N, -1)  # Flatten over packs_per_group

#             unpacked_values.append(masked)

#         # Stack the unpacked values and transpose to get correct order
#         # After stacking: Shape becomes (num_values_per_uint32, N, total_packed_values)
#         w_unpacked_stack = Tensor.stack(*unpacked_values, dim=0)
#         w_unpacked_group = w_unpacked_stack.permute(1, 2, 0).reshape(N, groups)  # Shape: (N, groups)

#         # Dequantize the unpacked weights
#         w_group = w_unpacked_group * scale_g + bias_g  # Shape: (N, groups)

#         # Extract the input activations for the current group
#         x_group = x_grouped[:, g, :]  # Shape: (M, groups)

#         # Perform matrix multiplication and accumulate the result
#         partial_output = x_group @ w_group.transpose()  # Shape: (M, N)
#         output += partial_output

#     return output

# ll = quantized_matmul_tg(Tensor(x), Tensor(w), Tensor(s), Tensor(b))
# ll = ll.flatten()
# print(ll.shape)
# print(ll.numpy())
# print(mx.quantized_matmul(mx.array(x), mx.array(w), mx.array(s), mx.array(b)))
# print(mx.quantized_matmul(mx.array(x), mx.array(w), mx.array(s), mx.array(b)).shape)