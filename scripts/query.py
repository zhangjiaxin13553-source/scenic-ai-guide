import os
import chromadb
from transformers import AutoTokenizer, AutoModel
import torch

# 配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

LOCAL_MODEL_PATH = os.path.join(SCRIPT_DIR, "bge-small-zh-v1.5")
HF_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 优先本地路径（Transformers老版本环境），不存在则用 HF 名称（新版环境）
if os.path.isdir(LOCAL_MODEL_PATH):
    MODEL_PATH = LOCAL_MODEL_PATH
else:
    MODEL_PATH = HF_MODEL_NAME
CHROMA_STORE = os.path.join(ROOT, "chroma_db")

TOP_K = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH).to(device)
model.eval()

def encode_query(text: str):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()

def search(query: str):
    client = chromadb.PersistentClient(path=CHROMA_STORE)
    coll = client.get_collection("luxun_know_base")
    vec = encode_query(query)
    res = coll.query(query_embeddings=[vec], n_results=TOP_K)
    return res

if __name__ == "__main__":
    print("=== 鲁迅知识库检索工具 | 输入exit退出 ===")
    while True:
        q = input("\n请输入提问：")
        if q.strip().lower() == "exit":
            break
        ret = search(q)
        print("-------- Top5 检索结果 --------")
        for i in range(TOP_K):
            print(f"\n【第{i+1}条】")
            print(f"文本片段：{ret['documents'][0][i]}")
            print(f"元数据：{ret['metadatas'][0][i]}")
            print(f"相似度距离：{ret['distances'][0][i]:.4f}")
