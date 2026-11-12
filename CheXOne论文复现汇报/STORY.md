# STORY.md — CheXOne 论文复现实验汇报

## ① 意图
- 受众：课题组组会 / 项目汇报（自己人，看得懂医学影像与 VLM）
- 核心目标：让听众相信——我们的评测链路是对的（View、Findings-BLEU 已复现），grounding / long-tail 低于论文是有可量化原因的；并记住：CheXOne 定位只在大结构上可用，迁移到 Track 8 小病灶不行
- 页数：13
- 调性：学术严谨 / 证据驱动 / 蓝白克制
- 内容边界：讲 5 个任务各自做什么、怎么做、结果、对比论文；不讲 RadEval 等已弃用路线；mAP 论文定义未公开，不强行对标

## ② 骨架
01 封面 / 02 总览（结论先行） / 03 统一评测设置 / 04 View / 05 Long-tail / 06 Phrase grounding 设置+结果 / 07 Phrase 细分 / 08 Abnormality 设置+结果 / 09 Abnormality 细分 / 10 Findings / 11 差距归因 / 12 迁移到 Track 8 / 13 结论与下一步
Hero：01、02、12（占比约 23%）

## ③ 大纲
| # | title | role | rhythm | layout | anti_pattern |
|---|---|---|---|---|---|
| 01 | 封面 | hero | peak | 全屏深蓝 + 论文式标题 | 禁止装饰小图、金色 |
| 02 | 5 个任务一览：2 复现、3 低于论文 | hero | peak | 图表+洞察（总表 70%+结论条） | 禁止等宽卡片 |
| 03 | 统一评测设置 | supporting | valley | 非对称双栏 + 流程节点 | 禁止纯文字罗列 |
| 04 | View classification | supporting | valley | 左标题+右内容 | 禁止 N 卡横排 |
| 05 | Long-tail | supporting | valley | 非对称双栏（图表 60%） | 禁止只写数字不给判断 |
| 06 | Phrase grounding | supporting | peak | 巨型数字+洞察 | 禁止 N 卡横排 |
| 07 | Phrase 细分 | supporting | valley | 双图表 | 禁止图表无洞察 |
| 08 | Abnormality grounding | supporting | valley | 左标题+右内容 | 同上 |
| 09 | Abnormality 细分 | supporting | valley | 图表+洞察 | 同上 |
| 10 | Findings generation | supporting | peak | 表格+洞察 | 禁止把不可比指标并排当结论 |
| 11 | 差距归因 | supporting | valley | 非对称双栏 | 禁止空泛“可能” |
| 12 | 迁移到 Track 8 | hero | peak | 巨型数字+图表 | 禁止卡片堆叠 |
| 13 | 结论与下一步 | supporting | transition | 左结论右行动 | 禁止空话 |

## 数据落点（数据+判断）
- View 0.988 vs 0.998，CI 重叠 → 链路正确
- BLEU 0.204 vs 0.215 → 生成链路正确
- MS-CXR 0.427 vs 0.608；instruct 时 45% 无框 → 问法影响大；出框后定位本身不差
- VinDr 0.250 vs 0.452；Q4 大病灶 / Cardiomegaly 0.577 → 差距来自样本构成与病灶大小
- Long-tail 0.757 vs 0.816；主动脉两类特异度 0.39/0.61 → 随机阴性样本含标签噪声（推测）
- Track 8 斜率 0.37/0.44 vs VinDr 0.61/0.66 vs MS-CXR 0.92/0.78 → 域内越近，框越跟着病灶走
