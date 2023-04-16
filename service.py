import bentoml
import torch
from bentoml.io import JSON
from transformers import AutoTokenizer, AutoModel


DEVICE = "cuda"
DEVICE_ID = "1"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class GPTRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("/Users/bangboom/Models/chatglm-6b", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("/Users/bangboom/Models/chatglm-6b", trust_remote_code=True).half().to('mps')
        self.model.eval()

    @bentoml.Runnable.method(batchable=False, batch_dim=0)
    def chat(self, data):
        query = data["query"]
        history = data.get("history", [])
        max_length = data.get("max_length", 2048)
        temeprature = data.get("temeprature", 0.7)
        response, history = self.model.chat(
            self.tokenizer,
            query,
            history=history,
            max_length=max_length,
            top_p=0.7,
            temperature=temeprature
        )
        torch_gc()
        return {
            "response": response,
            "history": history
        }


gpt_runner = bentoml.Runner(GPTRunnable, name="chat_glm", max_batch_size=10)

svc = bentoml.Service("chat_glm_service", runners=[gpt_runner])


@svc.api(input=JSON(), output=JSON())
def gpt(data):
    return gpt_runner.chat.run(data)
