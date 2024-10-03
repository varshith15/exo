import asyncio
import unittest

from exo.inference.shard import Shard
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.inference.mlx.sharded_inference_engine import MLXDynamicShardInferenceEngine


class PixtralShardTest(unittest.TestCase):
    def setUp(self):
        self.inference_engine = MLXDynamicShardInferenceEngine(HFShardDownloader())
        self.inference_engine1 = MLXDynamicShardInferenceEngine(HFShardDownloader())
        model_id = "varb15/hf-internal-testing-pixtral-12b"
        self.full_shard = Shard(model_id=model_id, start_layer=0, end_layer=39, n_layers=40)
        self.shard1 = Shard(model_id=model_id, start_layer=0, end_layer=12,
                            n_layers=40)
        self.shard2 = Shard(model_id=model_id, start_layer=13, end_layer=39,
                            n_layers=40)
        self.prompt = "<s>[INST]Describe the image in one sentence.\n[IMG][/INST]"
        self.img = "https://picsum.photos/id/237/400/300"
        self.max_tokens = 30

    async def _test_full_shard(self):
        resp, inference_state, _ = await self.inference_engine.infer_prompt("A", shard=self.full_shard,
                                                                            prompt=self.prompt, image_str=self.img)
        generated_tokens = [resp.item()]
        for _ in range(self.max_tokens):
            resp, inference_state, _ = await self.inference_engine.infer_tensor(
                "A",
                shard=self.full_shard,
                input_data=resp,
                inference_state=inference_state,
            )
            generated_tokens.append(resp.item())

        print("Response:\n", self.inference_engine.tokenizer.decode(generated_tokens))
        return generated_tokens

    async def _test_partial_shard(self):
        resp1, inference_state_1, _ = await self.inference_engine.infer_prompt("B", shard=self.shard1,
                                                                               prompt=self.prompt, image_str=self.img)
        resp2, inference_state_2, _ = await self.inference_engine1.infer_tensor("B", shard=self.shard2,
                                                                                input_data=resp1,
                                                                                inference_state=inference_state_1)
        generated_tokens = [resp2.item()]
        for _ in range(self.max_tokens):
            resp1, inference_state_1, _ = await self.inference_engine.infer_tensor("B", shard=self.shard1,
                                                                                   input_data=resp2,
                                                                                   inference_state=inference_state_2)
            resp2, inference_state_2, _ = await self.inference_engine1.infer_tensor("B" , shard=self.shard2,
                                                                                    input_data=resp1,
                                                                                    inference_state=inference_state_1)
            generated_tokens.append(resp2.item())
        print("Response:\n", self.inference_engine.tokenizer.decode(generated_tokens))
        return generated_tokens

    def test_inference_engine(self):
        full_generated_tokens = asyncio.run(self._test_full_shard())
        sharded_generated_tokens = asyncio.run(self._test_partial_shard())
        assert full_generated_tokens == sharded_generated_tokens


if __name__ == "__main__":
    unittest.main(verbosity=2)