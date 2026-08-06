import os
import json
import chromadb
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

# ==================== 全局配置 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

LOCAL_MODEL_PATH = os.path.join(SCRIPT_DIR, "bge-small-zh-v1.5")
HF_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 优先本地路径（Transformers老版本环境），不存在则用 HF 名称（Transformers新版环境）
if os.path.isdir(LOCAL_MODEL_PATH):
    MODEL_PATH = LOCAL_MODEL_PATH
else:
    MODEL_PATH = HF_MODEL_NAME
JSON_FOLDER = os.path.join(ROOT, "data", "processed")
CHROMA_STORE = os.path.join(ROOT, "chroma_db")
# 文本分块参数
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# 加载本地BGE模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH).to(device)
model.eval()

def get_embedding(text: str):
    """生成文本向量"""
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out = model(**inputs)
    vec = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()
    return vec

def slide_split(text: str):
    """滑动窗口分块"""
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + CHUNK_SIZE, length)
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def load_all_formatted_json():
    """只读取processed目录5类标准JSON，忽略raw文件夹"""
    json_files = ["venue.json", "work.json", "bio.json", "quote.json", "persona.json"]
    total_knowledge = []
    for file in json_files:
        file_path = os.path.join(JSON_FOLDER, file)
        if not os.path.exists(file_path):
            print(f"提示：{file} 不存在，跳过该类数据")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
            total_knowledge.extend(data_list)
    return total_knowledge

def main():
    # 初始化持久化向量库
    chroma_client = chromadb.PersistentClient(path=CHROMA_STORE)
    coll = chroma_client.get_or_create_collection(name="luxun_know_base")

    knowledge_data = load_all_formatted_json()
    print(f"成功加载标准化JSON条目总数：{len(knowledge_data)}")

    batch_ids = []
    batch_embeds = []
    batch_docs = []
    batch_meta = []

    for item in tqdm(knowledge_data, desc="JSON分块+向量化入库"):
        base_id = item["id"]
        content_text = item["content"]
        text_blocks = slide_split(content_text)

        for idx, block in enumerate(text_blocks):
            chunk_id = f"{base_id}_blk_{idx}"
            emb = get_embedding(block)
            # 完整存储Schema元数据，支持检索过滤
            meta_info = {
                "type": item["type"],
                "title": item["title"],
                "source": item["source"],
                "year": item.get("year", 0),
                "character_voice": item.get("character_voice", False),
                "venue_relevant": item.get("venue_relevant", False),
                "tags": ",".join(item.get("tags", []))
            }
            batch_ids.append(chunk_id)
            batch_embeds.append(emb)
            batch_docs.append(block)
            batch_meta.append(meta_info)

    # 批量写入向量库
    coll.add(
        ids=batch_ids,
        embeddings=batch_embeds,
        documents=batch_docs,
        metadatas=batch_meta
    )
    print(f"向量入库完成，总分块数量：{len(batch_ids)}")
    print(f"向量库存放路径：{CHROMA_STORE}")

if __name__ == "__main__":
    main()
