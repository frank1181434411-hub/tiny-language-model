# 0.1.0a2 wheel 构建与独立安装验收

2026-10-28 完成。Python 包版本 `0.1.0a2` 对应计划 Git 标签 `v0.1.0-alpha.2`。
本记录不代表标签或 Release 已发布。机器可读记录见 [JSON](wheel-alpha2-checks.json)。

## 构建产物

- 文件：[tiny_language_model-0.1.0a2-py3-none-any.whl](../dist/tiny_language_model-0.1.0a2-py3-none-any.whl)，60127 字节。
- [SHA-256 文件](../dist/tiny_language_model-0.1.0a2-py3-none-any.whl.sha256)：
  `947509a8677d1c20ec3fa3a24cc8e1e3d9e317529f3f62b317fae08e52ed33b2`。
- 使用已有 Python 3.12 环境和 setuptools 78.1.0，从当前源码实际构建；没有重命名旧包。
- ZIP CRC、RECORD 内容散列及大小全部通过；23 个 Python 源文件与源码一致，四份许可文件与仓库一致。
- 元数据版本为 `0.1.0a2`，包含 tiny_lm 与 dist-info，不包含语料、训练权重或运行缓存。
- 旧 `0.1.0` wheel 保留为历史构建记录。

构建命令（在仓库根目录，用项目环境的 Python）：

```bash
python -I -X utf8 -m pip wheel . --no-deps --no-build-isolation --no-index --no-cache-dir --wheel-dir dist
```

构建未采用隔离环境，仍有 setuptools 对现有许可配置的弃用提示；本次没有调整打包后端或许可配置。

## 独立安装及实际运行

在仓库外的 `D:/Python/alpha2-wheel-acceptance-20261028` 创建全新 venv，
`include-system-site-packages=false`。环境为 Windows 11、Python 3.12.10、CPU FP32。
在线获取依赖未完成后取消，改用本机 pip 缓存中的兼容 wheel 离线安装，未复用旧环境的 site-packages。
安装时正常解析依赖，未使用 `--no-deps`；torch 为 2.11.0+cpu、numpy 为 2.5.4、regex 为 2026.9.29、PyYAML 为 6.0.3。

- `pip check` 通过；模块与控制台两种入口都报告 `0.1.0a2`，帮助和配置检查通过。
- 使用自带 toy_corpus 和 smoke 配置，256 词表（零次 merge），完成 tokenizer 与训练/验证数据预处理。
- 训练到第 2 步，读取 checkpoint 恢复至第 4 步；独立验证 4 批、128 tokens，seed=42，loss=5.641384243965149。
- 从 `Hello` 生成 32 个新 token，因长度结束。四步随机小模型的输出不是故事质量验收。
- 另对 GELU 配置执行前向检查，确认使用 GELU 且输出有限；没有改动已有 SwiGLU 权重。
- 所有项目模块从新环境 site-packages 导入；七个流程命令通过审计钩子禁止读取原仓库，未触发违规读取。
- 安装来源 direct_url 的 SHA-256 与上述 wheel 一致，非 editable 安装；四份许可文件存在。

共 12 个命令成功。首轮外部验收脚本按 GBK 解码中文帮助时失败，CLI 本身退出码为 0；
为子进程设置 UTF-8 后重新完成验收，未修改项目 CLI。原失败记录保留在本地。
新演示输入及其散列记录在 JSON 中，不将其 loss 与历史不同输入的演示直接比较。

## 检查范围与记录位置

版本相关 CLI/import 测试本轮为 **34 passed**。此次只改版本与发布文档，不重复长时间训练或全量算法测试。
本记录不扩展到 Linux、GPU 或联网安装速度；跨设备源码复现仍见[独立记录](cross-device-checks.md)。

本地完整日志位于 `D:/Python/alpha2-wheel-acceptance-20261028/logs-v2/`，
构建检查位于仓库 `artifacts/alpha2-build-20261028/`；环境、缓存和训练产物不随 Git 提交。
新 wheel 和校验文件已加入精确的 Git 忽略例外，可与版本、发布文档一起提交。
