# 贡献指南

感谢你对 `cs2pickem` 的关注!本项目是一套**离线优先**的 CS2 Major Pick'em 预测工具链，欢迎通过 issue、PR 参与改进。

## 开发环境

需要 **Python 3.9+**。核心包零三方依赖，免安装即可运行：

```bash
# 免安装：直接用源码运行
PYTHONPATH=src python3 -m cs2pickem.cli demo

# 或安装为可编辑包（含开发依赖）
pip install -e ".[dev]"
```

可选加速依赖（缺失时自动回退纯 Python，不会崩溃）：

```bash
pip install -e ".[ml]"      # numpy / scikit-learn / xgboost / joblib 等
pip install -e ".[scrape]"  # requests / beautifulsoup4
pip install -e ".[viz]"     # matplotlib
```

> 详见仓库根目录的 `AGENTS.md`：默认主栈用 scikit-learn + XGBoost + joblib，神经网络分量默认保留纯 Python 实现（仅 `CS2PICKEM_ACCELERATED_MLP=1` 时才尝试 sklearn MLP）。

## 运行测试

提交前请确保单元测试全部通过：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

快速冒烟测试：

```bash
PYTHONPATH=src python3 -m cs2pickem.cli demo
```

## 代码约定

- **零依赖核心 + 优雅降级**：核心链路只用标准库；任何可选依赖缺失或导入失败时，必须回退纯 Python 实现而非崩溃。
- **无未来数据泄漏**：滚动特征、赛前 Elo、选手窗口、赔率/BP 合并都要按比赛日期截断，不得使用赛后信息。
- **测试驱动（TDD）**：新增或修改逻辑请附带测试，先写失败用例再实现。
- **确定性可复现**：切分、采样、模拟都应可控；报告需保留超参数、后端与数据来源以便审计。
- **市场信号克制**：真实赔率只做轻量修正；民调 proxy 只进报告，不当赔率使用。

## 提交规范

提交信息使用语义化前缀，例如：

```
feat: 新增 Elo 评分特征
fix: 修复退化的神经网络分量
docs: 更新 README 预测结果
test: 补充校准评估用例
```

## Pull Request 流程

1. Fork 并基于 `main` 创建特性分支。
2. 完成改动，确保 `unittest` 全绿、新逻辑有测试覆盖。
3. 按 PR 模板填写变更说明与检查清单。
4. 关联相关 issue（如有）。

## 数据与免责声明

本项目仅用于学习与研究。预测结果是赛前快照，不构成任何投注建议；请遵守所在地区相关法律法规。

## 许可

贡献的代码默认以本仓库的 [MIT 许可证](LICENSE) 授权。
