# dPLMProject
本项目用于蛋白质突变效应预测，包含多个基于深度学习和机器学习的模型。

## 📁 当前状态
目前上传了以下文件：

- `dPLM-GNN_predict.py`
- `dPLM-RFC.py`
- `dPLM-SVC.py`

其他内容仍在整理中，将会陆续补全。

## 📦 安装依赖（以 CUDA 12.1 为例）

安装以下库：`torch_geometric`, `torch`, `dgl`, `fair-esm`

```bash
pip3 install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install -c dglteam/label/cu121 dgl
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install torch_geometric
pip install fair-esm
