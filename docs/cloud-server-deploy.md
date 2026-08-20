# 在线体验 Demo 部署指南 —— 云服务器

把「鲁迅数字人 · 双模式对话」部署到一台**有公网 IP 的云服务器**，获得一个稳定的公网链接（`http://<公网IP>:7860`）。

---

## 0. 资源清单

| 资源 | 怎么获得 |
|------|---------|
| 代码 | `git clone`（服务器上拉取） |
| BGE 模型 + 向量库 + 知识库（「三件套」） | 本机 `pack_offline.py` 打成 zip，传到服务器 |
| DeepSeek API Key | 执行脚本时通过 `LLM_API_KEY` 环境变量注入（不写进脚本/仓库） |

---

---

## 1. 本机准备：打包「三件套」离线资源

在本机项目根目录执行（会把 BGE 模型 + 向量库 + 知识库打成单个 zip）：

```bash
cd scenic-ai-guide
python scripts/pack_offline.py
# 产出 releases/offline_resources.zip（约 100MB）
```

---

## 2. 上传资源 + 脚本到服务器

把 `offline_resources.zip` 传到服务器（用任意方式，`scp` 举例）：

```bash
scp releases/offline_resources.zip ubuntu@<服务器公网IP>:~/offline_resources.zip
```

代码不需要手动传 —— 部署脚本会自动 `git clone`。

---

## 3. 服务器上执行部署

SSH 登录服务器后，先下载部署脚本（或直接把本仓库的 `scripts/deploy_cloud.sh` 传上去）：

```bash
# 方式一：直接拉脚本
curl -O https://raw.githubusercontent.com/zhangjiaxin13553-source/scenic-ai-guide/main/scripts/deploy_cloud.sh

# 方式二：从本机 scp
#   scp scripts/deploy_cloud.sh ubuntu@<公网IP>:~/

# 执行（注意：把 sk-xxx 换成真实 DeepSeek Key）
LLM_API_KEY=sk-xxx bash deploy_cloud.sh ~/offline_resources.zip
```

脚本会自动完成：装 python3/git → clone 仓库 → 建 venv → 装 CPU 版 torch + 依赖 → 放置三件套 → 写 `.env` → 后台启动 Gradio。

---

## 4. 放行安全组端口

云服务器默认只开 22（SSH）等端口。需要在**云控制台 → 该实例 → 防火墙/安全组**里放行 `7860`（TCP）端口，否则公网访问不到。

---

## 5. 验证

1. 服务器本地探活：`curl http://127.0.0.1:7860` 返回页面即就绪
2. 浏览器打开 `http://<公网IP>:7860`
3. 试问「这个展厅主要展什么？」（讲解员模式）、「鲁迅先生，您为什么要弃医从文？」（数字人模式）
4. 首次加载约 1 分钟（BGE 模型加载），之后秒回

---

## 7. 常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| 公网打不开 | 安全组没放行 7860 | 见第 5 步，放行 TCP 7860 |
| `pip` 装依赖很慢 | 国内访问 PyPI 慢 | 脚本里 `pip install` 前加 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| `torch` 下载慢/失败 | 国内访问 pytorch.org 慢 | 改用镜像：`pip install torch==2.12.0 -i https://pypi.tuna.tsinghua.edu.cn/simple`（或用 `download.pytorch.org` 的国内镜像） |
| 重启服务器后 demo 没了 | `nohup` 进程不随开机自启 | 用 `systemd` 托管（见下），或手动重跑脚本 |
| 想更新代码 | 改了主仓库 | 服务器上 `git -C $HOME/scenic-ai-guide pull`，然后重跑 `bash deploy_cloud.sh`（会重新装依赖并启动） |
| 端口被占用 | 7860 被别的进程占用 | `APP_PORT=7861 bash deploy_cloud.sh ...` 换端口，并同步放行新端口 |


