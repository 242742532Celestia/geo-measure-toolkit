---
name: geo-measure-toolkit
description: |
  GEO（生成式引擎优化）实验的测量与判读工具链：功效/MDE 反解、六条统计 lint 判据（L1–L6）、
  处理臂泄漏分 L(T,Y) 同义反复审计、prompt 冻结与逐位校验、候选特征抽取（不出分）、
  以及自由文本改写的信息保持门 G1–G9。纯标准库，无 numpy/scipy/matplotlib。
  Use when the user asks「这个效应量报得出来吗」「多少样本才够」「算一下 MDE」
  「这个 null 站得住吗」「审一下这批实验的统计」「审稿人说我统计有问题」
  「这个处理臂是不是把结果自己写进去了」「相关是不是被污染了」
  「这次改写有没有编造信息」「G1–G9 过了没有」「抽出来的东西会不会偷偷变成分数」
  「prompt 冻结了没有 / 能重建出同一批臂吗」;
  triggers on "power analysis", "minimum detectable effect", "statistical lint",
  "tautology audit", "information preservation gate", "prompt freezing".
  Do NOT use for: 纯描述性统计、给已完成分析改写措辞、论文语言润色、
  以及任何需要真实实验产出才能跑的分析（那些会明确报 needs-project）。
descriptions:
  zh-Hans: "GEO 实验的测量与判读工具链：功效/MDE 反解、统计 lint 判据 L1–L6、泄漏分 L(T,Y) 审计、prompt 冻结、候选特征抽取（不出分）、信息保持门 G1–G9。纯标准库。"
  en: "Measurement and judgment toolchain for GEO experiments: power/MDE inversion, statistical lint rules, tautology audit, prompt freezing, score-free feature extraction, information-preservation gate."
displayNames:
  zh-Hans: "GEO 测量与判读"
  en: "GEO Measurement & Judgment"
---

# geo-measure-toolkit — GEO 实验的测量与判读

## 定位

不主张「统计写得不好」，而主张**可报告性是可计算、可形式化、可自动检查的**。
每个结论都落到一个**数字**上：给定一篇论文报出的数字，反解出它真实能检出多小的效应；
给定一个填好的结果卡，跑一组形式化判据。

**工具不在本 skill 里。** 工具链是独立仓库（`geo-measure-toolkit`），
拷一份进 skill 目录就造出第二个事实来源 —— 两份会漂移，且漂移是**静默的**：
论文里的 MDE 由项目那份算出，审稿时跑的却是 skill 里那份过时的。

⇒ **只引用，不复制。** 下面每个工具都给仓库路径；需要跑就 clone 仓库，
或指向已安装的那一份。

```bash
git clone https://github.com/242742532Celestia/geo-measure-toolkit.git
```

仓库地址：<https://github.com/242742532Celestia/geo-measure-toolkit>
（换账号用 `python scripts/set_repo_url.py --owner <账号>` —— 本文件、`CITATION.cff`、
README badge 三处一起改，脚本已经覆盖 SKILL.md。）

---

## 工具清单

| 工具 | 仓库路径 | 回答什么问题 | 自检 |
|---|---|---|---|
| 功效反解内核 | `tools/statpower.py` | 这个效应量报得出来吗？多少样本才够？ | 33 项 |
| 统计 lint 判据 | `tools/geo_stat_lint.py` | 六条判据 L1–L6：这个 null 站得住吗？ | 26 项 |
| 同义反复审计 | `tools/tautology_audit.py` | 这个处理臂是不是把结果自己写进去了？ | 30 项 |
| prompt 冻结 | `tools/freeze_prompts.py` | prompt 冻结了吗？能重建出同一批臂吗？ | 83 项 |
| 候选特征抽取 | `tools/geo_page_features.py` | 抽出来的东西会不会偷偷变成分数？ | 61 项 |
| 信息保持门 | `tools/info_gate.py` | 这次改写有没有编造信息？G1–G9 过了没有？ | 49–53 项 |
| 构念效度 | `examples/11_construct_validity.py` | 测的是不是我要测的那个构念？ | 22 项 |
| 配对分析 | `examples/19_paired_analysis.py` | 配对怎么做才对（Holm 校正、退化语义）？ | 81 项 |
| 方差分解 | `examples/14_stage_a_variance.py` | σ_b / σ_e 怎么分、样本量怎么定型？ | 67 项 |
| 噪声底噪 | `examples/24_e4_noise_floor.py` | 方差分解与 MDE 的算术下界 | 13 项 |
| 序贯停止 | `examples/23_e4_sequential.py` | 序贯停止条件（预注册的一部分） | 16 项 |
| 改写器噪声 | `examples/20_rewriter_noise.py` | 改写器引入了多少噪声？ | 24 项 |
| 引文噪声 | `examples/10_citation_noise.py` | 引文率上的 run-to-run 噪声 | 10 项 |
| 基础设施/方法分层 | `examples/13_infrastructure_survey.py` | 文献是「都能」还是分层不同？ | 17 项 |

自检跑法：

```bash
python run_selftests.py            # 全跑
python run_selftests.py --list     # 只看清单
python run_selftests.py --only statpower   # 只跑某个工具
```

---

## 四种判定态：缺席 ≠ 失败

判读任何工具输出前先记住这张表。**把「环境缺一块」报成「工具坏了」，
处置方向是反的** —— 前者要补环境，后者要改代码。

| 情形 | 工具表现 | runner 判定 | 该做什么 |
|---|---|---|---|
| 前置缺失 | 结项行声明「前置缺失」 | `SKIP` | 补环境 |
| 声明过的缺席 | 少跑几项，但打了 `[SKIP]` 说明原因 | `SKIP` | 补环境 |
| 静默失检 | 少跑几项，**什么都没说** | `FAIL` | **修工具** |
| 崩了 | 没有结项行 | `FAIL` | **修工具** |

实测比登记**多**也不算失败 —— 那是登记表过期，重跑
`python tools/build_manifest.py` 更新即可。

---

## 用法要点

### 1. 判读一个报出来的数字（最常见）

```bash
python tools/statpower.py selftest                          # 先确认工具本身是绿的
python tools/statpower.py mde-prop --p1 0.0564 --n1 448 --n2 448
python tools/statpower.py n-for-prop --p1 0.0564 --p2 0.0919
```

要点：**先反解 MDE，再看论文报的效应量是不是比 MDE 大**。
比 MDE 小 ⇒ 那个设计报不出这个效应 ⇒ 结论站不住。
`geo_stat_lint.py` 把这条判据固化成 L1–L6，可直接跑在一张填好的结果卡上：

```bash
python tools/geo_stat_lint.py run cards/*.json     # cards 是结果卡 JSON，支持 glob
python tools/geo_stat_lint.py explain L2            # 解释某条判据
```

### 2. 审一个处理臂

```bash
python tools/tautology_audit.py --spec <spec.json>
python tools/tautology_audit.py --builtin           # 跑内置的阳性对照
python tools/tautology_audit.py --survey            # L 判据的横向适用性
```

看的是**泄漏分 L(T,Y)**：处理臂如果「机械地」改写了因变量自带的成分，
那不是效应，是自证。

### 3. 审一次改写

```bash
python tools/info_gate.py --selftest
```

门只判**诚实**（有没有引入源文本里没有的事实性内容），不判改写好不好。
G1–G9 的依据来自已发表的 GEO 改写研究，模块头注里列了出处。

### 4. 冻结 prompt 以便日后重建配对

```bash
python tools/freeze_prompts.py --export
python tools/freeze_prompts.py --verify
```

**配对分析必须按冻结输入重建** —— 否则「配对」只是两批不同输入的巧合。

---

## 边界（什么时候不要用这个 skill）

* 纯描述性统计、给已完成的分析改写措辞 —— 不是这里的事。
* 需要真实实验产出的分析（`14_stage_a_variance` / `19_paired_analysis`）：
  它们带**合成数据退化路径**——真实产出不在场时用合成数据验代码路径，
  并**出声**说明用的是合成数据。合成数据证明的是代码路径，
  **不证明真实数值**；真实数字仍然只能由真实数据说话。
* `08_reproduce_geo_bench.py` **没有自检入口**（验证缺口）—— 依赖它之前先人工确认。
* 候选特征抽取器**只出候选预测变量**，不是分数、不是权重、不是档位。
  要打分请用别的工具。

---

## 可选外部参考模块

`info_gate.py` 可以拿上游参考实现做**逐位比对**。那个模块是**可选**的：

```bash
export GEO_CORE_PATH=/path/to/geo_core
```

不设 → 用内置兜底常量，比对类断言报 `[SKIP]`（声明过的缺席，不是失败）；
设了 → 比对会跑，自检从 49 项变成 53 项。

⚠️ `info_gate_selfcontained_check.py` 会**临时改名**该目录再跑一遍
（这是「不依赖包外」唯一的可执行证据）。**别在跑的半路强杀进程** ——
改名与恢复都在 `finally` 里，强杀会留下 `.__hidden_by_selfcheck` 残留。
