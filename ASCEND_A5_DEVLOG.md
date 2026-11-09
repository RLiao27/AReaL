# AReaL v2.1.0 Ascend A5 开发记录

更新日期：2026-10-29（Asia/Shanghai）

> 本文按时间保留开发过程，其中“未提交的代码改动”、早期验证结果和后续计划是当时的历史快照，不代表当前代码或最终结论。当前范围、最终实验数据和已知限制见
> [Ascend A5 适配说明](docs/zh/tutorial/ascend_a5_v21.md)。

## 基线和工作环境

- 开发仓库：宿主机 `/home/areal-v2-a5/AReaL`，容器内 `/workspace/AReaL`。
- 仓库新克隆自 `https://github.com/RLiao27/AReaL.git`，检出 `v2.1.0`，基线提交
  `ecc8b0e4dfb4e3965f67121a5c87866a07c390ae`。
- 从该基线创建分支 `feature/ascend-a5-bf16`；当前改动未提交。
- 参考仓库：宿主机 `/home/lr/AReaL` 的 `origin/ascend-950-dev`，参考提交
  `d3a69c14cb6e74a17be43e7a8b3b27b23bde416f`。
- 容器 `areal-v2-a5`，镜像 `de39a00c8e05`；宿主机 `/home/areal-v2-a5` 挂载为 `/workspace`。容器启用了
  privileged 和 host network。
- 镜像环境：Python 3.12.13、PyTorch 2.10.0+cpu、torch-npu 2.10.0.post2、Transformers 5.5.4、vLLM
  0.23.0+empty、vLLM Ascend 0.23.0rc2.dev107+geaefc536d、MindSpeed 0.16.0、CANN 9.0.1。
- 镜像内 NumPy、SciPy 文件及 torchvision 二进制不匹配，临时在 `/workspace/python-overlay` 安装了 NumPy
  1.26.4、SciPy 1.13.1、torchvision 0.25.0+cpu。运行测试时显式设置
  `PYTHONPATH=/workspace/python-overlay:/workspace/AReaL`，缓存放在
  `/workspace/triton-cache`。MindSpeed 声明 NumPy \<=1.26.0，此依赖约束冲突尚未解决；不要把覆盖层视为最终环境方案。

## 当时的代码改动（历史快照，后续已清理）

- `areal/engine/fsdp_engine.py`：NPU 上初始化 Qwen3.5 时加载 GDN 适配；暂时拒绝 SP > 1。
- `areal/engine/fsdp_utils/fla_patch_npu.py`：以 Ascend 分支为参考迁入 Qwen3.5 GDN、RMSNorm、RoPE 和
  MLP 适配；为 dense/MoE 解码层传递打包边界；打包数据不能安全降级时显式报错。
- `areal/engine/fsdp_utils/gdn_triton/`：迁入 NPU causal-conv1d 实现。
- `areal/engine/megatron_utils/triton_l2norm_patch.py`：迁入 MindSpeed GDN 所需 L2 norm 补丁。
- `tests/test_qwen35_npu_packing.py`：新增打包结果一致性和带填充段的反向梯度测试。
- 上述代码为实验性工作，不是已经验证的完整适配；未迁移 vLLM BF16 权重同步，未开始 MXFP8。

## 验证记录和未解决问题

- 容器内 8 张 Ascend 950 可见，NPU 张量基本运算通过。
- 小型 Qwen3.5 GDN BF16 前向和反向（32 token）梯度有限；完整小型混合模型打包与独立样本前向比较通过。
- 单卡 FSDP 小模型可初始化，但实际训练步骤的梯度范数为 NaN，优化器没有更新。
- 测试 `tests/test_qwen35_npu_packing.py` 当前结果为 1 passed、1 failed。输入长度
  256、`cu_seqlens=[0,16,32,256]` 时 GDN 反向产生非有限梯度；相同长度单序列 `[0,256]` 和两个等长序列 `[0,128,256]`
  的独立 GDN 测试梯度有限。异常检测指向 MindSpeed `ChunkGatedDeltaRuleFunctionBackward`；根因尚未确认。
- 测试辅助脚本
  `/workspace/test_gdn_smoke.py`、`/workspace/test_gdn_padded.py`、`/workspace/test_qwen_model.py`、`/workspace/fsdp_smoke.py`
  及日志位于仓库外的工作目录。
- 测试命令：`docker exec -e PYTHONPATH=/workspace/python-overlay:/workspace/AReaL -e TRITON_CACHE_DIR=/workspace/triton-cache areal-v2-a5 python -m pytest -q tests/test_qwen35_npu_packing.py`（在宿主机运行）。

## 后续工作

1. 修复变长打包 GDN 的非有限反向梯度，验证数值和梯度与独立序列一致。
1. 验证 FSDP 优化器成功更新，以及多卡、真实 Qwen3.5 权重和数据路径。
1. 迁移并验证 vLLM Ascend BF16 推理与 FSDP 权重同步，跑通端到端 RL。
1. 基础 BF16 路径完成后，再参考 `ascend-950-dev` 实现和验证 MXFP8。

## Qwen3-8B 非 GDN 路径（2026-10-29）

- 从服务器现有权重复制模型到 `/workspace/models/Qwen3-8B`。单卡 vLLM Ascend 0.23 BF16 加载和生成通过。
- `areal/trainer/rl_trainer.py`：v2 全参数训练显式选择 `weight_update_mode=disk` 时，使用磁盘权重更新；未指定
  disk 时保留 AWEX 选择。
- `areal/engine/vllm_ext/areal_vllm_server.py`：兼容 vLLM 0.23 的 API
  工具导入路径，同时保留旧版路径。容器内导入通过。
- 双卡 FSDP 加单卡 vLLM 的 v2 RL 启动测试使用 `/workspace/run-qwen3-v2.sh` 和
  `/workspace/math-smoke-data`；首次启动因 vLLM 导入路径失败，修复后尚未重新验证完整闭环。
- 测试期间发现服务器 8 卡均有其他运行中的工作，已停止本次测试的进程，未停止其他任务。等待卡空闲后重新测试至少两步，检查实际参数更新和推理权重同步。
- 日志：`/workspace/vllm-smoke.log`、`/workspace/qwen3-v2-smoke.log`。相关开发文件在服务器和本地仓库保持一致；代码仍未提交。
- NPU 示例 `examples/math/gsm8k_grpo_npu.yaml` 的权重更新方式设为 `disk`。v2 控制器仍需在启动配置中显式启用；disk
  端到端流程待空闲 NPU 验证。

## Qwen3-8B v2 复测（2026-10-29 晚）

- 启动时 8 卡空闲；数据服务、双卡 FSDP worker 和 vLLM API 服务启动，vLLM 旧版导入路径问题已越过。
- 推理 worker 因 `ModuleNotFoundError: No module named 'acl'` 退出。原因是启动脚本覆盖了镜像的
  `PYTHONPATH`，丢失 CANN Python 路径。已修正 `/workspace/run-qwen3-v2.sh` 和
  `/workspace/run-qwen3-v2-verify.sh`，保留原有 `PYTHONPATH`；容器内 `import acl.rt` 通过。
- 启动后另一组工作占用了全部 8 卡；已停止本次验证的进程。还未完成训练步骤或 disk
  权重同步，待获得独占可用卡后重跑。复测日志：`/workspace/qwen3-v2-verify.log` 和 `/workspace/rl-verify/logs/`。

## Qwen3-8B v2 BF16 冒烟验证通过（2026-10-29 晚）

- 单机 3 卡：FSDP 数据并行 2 卡、vLLM Ascend 推理 1 卡；使用 16 条合成数学题、每题 2 个采样、2 步 GRPO 和 disk 权重更新。
- 第一步优化器报告成功，但该批次组内奖励相同，梯度范数为 0；第二步优化器更新成功，梯度范数 4.5571，样本奖励存在差异。
- disk 权重更新 v1、v2 均返回 `ok`，分别约 17.1 秒和 15.8 秒；第二步使用第一步更新后的推理版本。进程正常退出并释放 NPU。
- 修复 vLLM 服务缺少 `/update_weights_from_disk` 接口，复用已有权重加载实现；修复 v2
  训练控制器在同步返回错误时仍继续训练的问题，新增成功和失败分支测试。
- 测试：`tests/v2/training_service/test_weight_update_failure.py` 与
  `tests/v2/weight_update/test_wu_controller.py` 共 13 项通过；相关文件 Ruff 检查通过。
- 启动脚本 `/workspace/run-qwen3-v2-run4.sh`，主日志 `/workspace/qwen3-v2-run4.log`，服务日志
  `/workspace/rl-run4/logs/root/qwen3-a5-v2-run4/trial0/`。该结果是小规模冒烟验证，尚未验证长时间训练、多机、Qwen3.5
  或 MXFP8。

## GSM8K 正式数据训练（2026-10-30）

- 复用本机 Hugging Face 缓存，训练集 7473 条、测试集 1319 条；预处理数据位于 `/workspace/real-math`。
- 原始 Qwen3-8B 在完整测试集上贪心解码、最多生成 1024 token，math_verify 评分为 792/1319（60.05%）；结果保存在
  `/workspace/gsm8k-baseline.json`，评估脚本 `/workspace/eval-real-math.py`。
- v2 disk 同步增加遵循 `clear_checkpoint_after_load` 的非 LoRA 临时快照清理：所有推理 worker
  成功加载后清理，失败时保留；默认服务协议保留旧行为。单元测试 25 项通过，真实 A5 两步试验后临时目录为 4 KB。
- 整轮训练使用 `/workspace/run-gsm8k-training-r2.sh`，日志 `/workspace/gsm8k-training-r2.log`；4 卡
  FSDP、1 卡推理，batch size 64，每题 4 次采样，最大生成 1024 token，rollout 并发 16，每 25 步及 epoch
  结束保存正式检查点。训练仍在运行，最终评估尚未完成。
- 先前并发 128 的尝试在暂停推理时发生 HTTP 连接池超时，已停止该次训练；当前以并发 16 继续验证。短验证完成后关闭过程曾滞留，已清理其进程树。
- `/home/areal-v2-a5/eval-after-training.sh` 在整轮训练完成后等待空闲 NPU
  5，并对最终检查点运行同一测试集评估；进度和对比结果写入 `/home/areal-v2-a5/gsm8k-final-status.txt`，详细评估日志写入
  `/home/areal-v2-a5/gsm8k-final-eval.log`。
- 训练已完成 116/116 步，最终检查点 `epoch0epochstep115globalstep115`；自动评估输出
  `/workspace/gsm8k-final.json` 为 0/1319，输出为重复换行且全部达到生成长度上限，问题根因待定位。原始模型基线 792/1319 中有
  1211 条达到长度上限，评分可能将中间数字当作答案，不能当作可靠 GSM8K 准确率。
- 评估进程结束太快时监控脚本误报没有结果；已根据 JSON 修正 `/workspace/gsm8k-final-status.txt`，不影响原始评估文件。

## Qwen3 训练退化修复与严格评估（2026-10-30）

- Qwen3 SDPA 打包时传入全空 attention-mask 字典会跳过 Transformers 的序列边界掩码，导致后一个样本可见前一个样本。改为对 Qwen3
  SDPA 传 `None`，由模型根据 position IDs 建立掩码；新增 FSDP 批次到小模型 logits 的隔离回归测试。
- A5 两步对照中，首步 rollout 和训练 logprob 的平均绝对差由旧实验约 2.08 降至
  0.0082。该问题是训练退化的明确风险，最终退化是否完全由它造成尚未单独证明。
- v2 数据代理此前丢弃顶层 `chat_template_kwargs`；已修复传递，实测 Qwen3 的 `enable_thinking=False` 生效。新增严格
  GSM8K 评分，只比较最后一个完整 boxed 答案，训练和评估使用同一规则。
- 修正后的 512-token 贪心基线：官方测试集 1208/1319（91.58%），正常结束 1294 条；从训练集种子 42 留出的 256 题验证集基线
  243/256（94.92%）。
- 新实验训练 7217 条、验证 256 条；4 卡 FSDP、1 卡 vLLM，batch size 64、4 次采样、学习率 1e-6、最多 512 新 token，并发
  16。启动脚本 `/workspace/run-gsm8k-strict-training.sh`，训练日志
  `/workspace/gsm8k-strict-training.log`。
- 每 25 步和最后保存检查点，由 `/workspace/supervise-gsm8k-strict.sh` 在 NPU 5
  空闲时使用验证集评估各检查点，训练完成后评估最终检查点的官方测试集；进度与结果位于 `/workspace/gsm8k-strict-status.txt`。
- 已删除旧 run3/run4 的四份临时同步权重，释放约 64 GB；正式检查点和历史问题证据保留。合并回归测试 40 项通过、3 项硬件测试未运行，Ruff 检查通过。
- 第 25 步检查点的 256 题验证集结果为 244/256（95.31%），与原始模型的 243/256 接近；训练仍在进行，官方测试集结果待最终检查点。
- 新实验已正常完成 112/112 步；第 50、75、100、112 步验证结果分别为 246、247、248、248 / 256。最终模型官方测试集为
  1223/1319（92.72%），相同严格评分下原始模型为 1208/1319（91.58%），提高 1.14 个百分点。逐题比较为 47 题由错变对、32
  题由对变错；最终模型没有空白输出，1294 条正常停止、25 条触及长度上限。
- 训练日志 `/workspace/gsm8k-strict-training.log`，基线及最终评估结果
  `/workspace/gsm8k-strict-baseline.json`、`/workspace/gsm8k-strict-final.json`；最终权重在
  `/workspace/gsm8k-strict-training/checkpoints/root/qwen3-8b-gsm8k-strict-v1/epoch1/default/epoch0epochstep111globalstep111`。本结果仅验证
  Qwen3-8B、FSDP、vLLM Ascend、BF16 和单机 disk 同步，Qwen3.5/GDN、MXFP8 未完成验证。
- 按适配范围清理：未验证的 Qwen3.5/GDN 补丁及 GSM8K 实验专用训练、评估和评分代码移至仓库外
  `a5-experiment-archive/`，Qwen3.5 的自动补丁入口已移除。评估脚本的新容器路径为
  `/workspace/a5-experiment-archive/examples/math/gsm8k_qwen3_eval.py`；训练启动脚本已更新到归档路径，GSM8K
  结果和权重保持原样。
