from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
import asyncio
import numpy as np


# An inference engine should work the same for any number of Shards, as long as the Shards are continuous.
async def test_inference_engine(inference_engine_1: InferenceEngine, model_id: str):
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained(model_id)
#   prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

# what is the full name of the current president of the USA and give some history?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
  prompt = "tell me about the origins of hip hop:"
  ans = []
  resp_full, inference_state_full, _ = await inference_engine_1.infer_prompt("A", shard=Shard(model_id=model_id, start_layer=0, end_layer=31, n_layers=32), prompt=prompt)
  ans.append(resp_full.item())
  for _ in range(2):
    resp_full, _next_inference_state_full, _ = await inference_engine_1.infer_tensor(
      "A",
      shard=Shard(model_id=model_id, start_layer=0, end_layer=31, n_layers=32),
      input_data=resp_full,
      inference_state=inference_state_full,
    )
    ans.append(resp_full.item())
    
  
  print(tokenizer.decode(ans))
  print(ans)

# # #   pp = 15
# # #   resp1, inference_state_1, _ = await inference_engine_1.infer_prompt("B", shard=Shard(model_id=model_id, start_layer=0, end_layer=pp, n_layers=32), prompt=prompt)
# # #   resp2, inference_state_2, _ = await inference_engine_2.infer_tensor(
# # #     "B",
# # #     shard=Shard(model_id=model_id, start_layer=pp + 1, end_layer=31, n_layers=32),
# # #     input_data=resp1,
# # #     inference_state=inference_state_1,
# # #   )
# # #   resp3, inference_state_3, _ = await inference_engine_1.infer_tensor(
# # #     "B",
# # #     shard=Shard(model_id=model_id, start_layer=0, end_layer=pp, n_layers=32),
# # #     input_data=resp2,
# # #     inference_state=inference_state_2,
# # #   )
# # #   resp4, _inference_state_4, _ = await inference_engine_2.infer_tensor(
# # #     "B",
# # #     shard=Shard(model_id=model_id, start_layer=pp + 1, end_layer=31, n_layers=32),
# # #     input_data=resp3,
# # #     inference_state=inference_state_3,
# # #   )

# # #   assert np.array_equal(resp_full, resp2)
# # #   assert np.array_equal(next_resp_full, resp4)


import tinygrad
import os
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine

if __name__ == "__main__":
    # model_id = "TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R"
    model_id = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
    tinygrad.helpers.DEBUG.value = int(os.getenv("TINYGRAD_DEBUG", default="0"))
    asyncio.run(
    test_inference_engine(
    TinygradDynamicShardInferenceEngine(HFShardDownloader()),
    model_id,
    )
    )


# from transformers import LlavaForConditionalGeneration, AutoProcessor
# from PIL import Image
# import torch

# model_id = "hf-internal-testing/pixtral-12b"
# model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
# processor = AutoProcessor.from_pretrained(model_id)

# IMG_URLS = [
#     "https://picsum.photos/id/237/400/300",
#     "https://picsum.photos/id/231/200/300",
#     "https://picsum.photos/id/27/500/500",
#     "https://picsum.photos/id/17/150/600",
# ]
# PROMPT = "<s>[INST]Describe the image.\n[IMG][/INST]"

# inputs = processor(images=IMG_URLS[0:1], text=PROMPT, return_tensors="pt")
# # print(inputs)
# # generate_ids = model.generate(**inputs, max_new_tokens=500)
# generate_ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"][0], attention_mask=inputs["attention_mask"], max_new_tokens=30)
# ouptut = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# print(ouptut)

# import torch
# from transformers import LlavaForConditionalGeneration, AutoProcessor

# model_id = "hf-internal-testing/pixtral-12b"
# hf_model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# processor = AutoProcessor.from_pretrained(model_id)

# IMG_URLS = [
#     "https://picsum.photos/id/237/400/300",
#     "https://picsum.photos/id/231/200/300",
#     "https://picsum.photos/id/27/500/500",
#     "https://picsum.photos/id/17/150/600",
# ]
# PROMPT = "<s>[INST]Describe the images in one sentence.\n[IMG][/INST]"

# inputs = processor(PROMPT, IMG_URLS[:1], return_tensors="pt")

# kp = hf_model.forward(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"][0], attention_mask=inputs["attention_mask"])

# print(kp)
