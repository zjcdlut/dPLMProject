# dPLMProject
本项目旨在通过蛋白质大语言模型提取的差异向量（dPLM）对蛋白质突变效应进行预测。模型结合了序列嵌入与三维结构信息，构建图神经网络（GNN）、多层感知机（MLP）、随机森林（RF）与支持向量机（SVM）等方法，用于分类突变是否具有功能性影响，广泛适用于酶工程与致病变异分析。

## 📁 当前状态
目前上传了以下文件：

- `dPLM-GNN_predict.py`
- `dPLM-RFC.py`
- `dPLM-SVC.py`

其余文件与数据处理流程正在整理中，后续将持续更新。

## 📦 安装依赖（以 CUDA 12.1 为例）

安装以下库：`torch_geometric`, `torch`, `dgl`, `fair-esm`

```bash
pip3 install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install -c dglteam/label/cu121 dgl
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install torch_geometric
pip install fair-esm
