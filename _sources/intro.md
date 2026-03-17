---
cover: https://fasterai-picgo.oss-cn-beijing.aliyuncs.com/20260119194153.png
create_date: 2026.01.01
---

# 动手学大语言模型：写给程序员的手搓LLM实战指南

## 自序

我们正身处一个被大语言模型深刻重塑的时代。一夜之间，能够理解并生成复杂文本、编写代码、进行推理的智能体不再是科幻构想，而是触手可及的现实。从能对话、能创作的智能助手，到自动生成代码、优化程序的AI程序员，再到深入科研、医疗、金融等领域的专业工具，大模型正在以前所未有的速度拓展能力的边界，并重新定义“智能”的疆域。在这个技术浪潮之巅，掌握大模型的原理与实践，已成为程序员面向未来的关键竞争力。

然而，面对日新月异的技术、层出不穷的论文与框架，许多开发者感受到的是“知识爆炸”带来的焦虑与迷茫。从复杂的数学公式到浩如烟海的工程细节，传统的AI学习路径漫长而陡峭，让人望而生畏。我们是否必须通晓所有理论，才能一窥门径？我们是否注定在追赶中疲惫，却难以触及核心？

上面提到的这些困境，我都亲身经历过，也深深体会过转型的阵痛。虽然我毕业于北大，自诩不算太笨，但在学习AI的过程中，也曾屡次想要放弃。经过不少艰难的学习和反复的摸索，我才终于入门，这期间踩过很多坑，也浪费了不少时间。这段经历让我真切地意识到，一条正确、高效、为程序员量身打造的大模型学习路径是多么重要。

于是我不禁思考：若能以今日之经验，给过去的自己提一些学习建议，该用什么方法才能提前一两年掌握AI，在学习 AI 的过程中少走些弯路？**我认为最好的方式就是理论结合代码实战，把复杂问题简单化，让大模型的学习曲线变得平缓。正因如此，我结合自身经验着手编写这本《动手学大语言模型》。**

我坚信，既然这些知识既然能解答我的困惑，也定能为你点亮一盏灯。引用奈尔斯的一句话：“我们就像乞丐一样，试图告诉其他乞丐我们在哪里找到了面包。”这也正是我创作这本《动手学大语言模型》的初心。倘若我的经验能让你的AI学习之路走得更轻松一些，我将倍感荣幸。

> **愿为清风，拂你三寸坦途；愿作微光，映你半步前行。**

### 本书的理念

**本书深信：理解根植于创造，精通发端于实践**。我尤为认同理查德·费曼的箴言——“凡我不能创造的，我就不能理解。”对注重实效的程序员而言，最深层的理解并非来自被动的阅读与聆听，而是来自主动的构建与调试。

因此，本书摒弃了从数学到原理的冗长铺垫，选择了一条更直接、更务实的路径：“**从零开始，手搓源码**”。我们将大模型这个看似庞然大物的“黑箱”，拆解为一系列可理解、可亲手实现的模块。就像搭积木一样，本书会带领你从最基础的组件开始，逐步构建出RNN、Transformer，乃至GPT等经典大模型架构。

我们的目标是：让你在动手实现的过程中，自然而然地掌握注意力机制、位置编码、层归一化等核心概念；让你在调试与优化的实践中，深刻体会模型训练的技巧。

**我们追求快速掌握“最少必要知识”，直击要害，绕过不必要的复杂，让你能用最短的时间，搭建起关于大模型的坚实认知框架，并掌握其实现的底层细节。**

### 本书的特色

本书的主要目标是帮助读者通过动手实践的方式快速地学懂LLM。为了实现这种“动手学”的理念，我设计了一系列的实战任务串联大模型的技术演进脉络，并运用大量视觉语言，使用数百张插图帮助读者对LLM学习过程中的主要概念和流程建立直观认识，降低理解的难度。

通过这种【任务驱动+图解式】的叙事方法，我们希望帮你轻松踏上通往这个令人振奋且可能改变世界的领域的旅程。

**耗尽心神、亲手制作的数百幅全彩插图，给你提供极致视觉化呈现，让所有的技术难点都可视化、易理解。**

![图片.png](https://fasterai-picgo.oss-cn-beijing.aliyuncs.com/%E5%9B%BE%E7%89%87.png)

### 本书面向的读者

本书是为这样的你准备的：

- 具备Python编程基础，渴望深入AI领域但被理论门槛所阻的程序员。
- 已在使用大模型API或框架进行应用开发，希望洞悉其底层原理以提升竞争力的开发者。
- 厌倦了浮于表面的概念介绍，渴望通过“做中学”来获得扎实技能的技术实践者。
- 希望在面试中能对模型细节侃侃而谈，从而抓住高薪机遇的求职者。

为帮助你高效学习，每个章节将提供**可运行的完整代码、明确的实验步骤，以及围绕代码展开的原理解读**。我希望你准备好开发环境，跟随本书的节奏，亲自运行、修改、甚至尝试重写这些代码——真正的理解，从动手开始。

### 本书不面向的读者

同时，明确本书的边界也同样重要：

- 本书不会系统性地讲授机器学习或深度学习的全部理论基础。
- 本书不会详细对比评测Hugging Face、LangChain等各类上层应用框架的使用技巧。
- 本书不会追踪并详解每一篇最新顶会论文的前沿细节。
- 本书不会涉及大规模分布式训练、极致推理优化等重型工业级工程议题。

**本书聚焦于通过“手搓源码”来理解大模型的核心架构与训练逻辑。完成本书的实践后，你将具备足够的能力和信心，去更自如地阅读论文、更深入地使用框架，并为你自己的原创性项目打下坚实基础。**

### 让我们动手开始吧

当下，调用一个强大的大模型API或许只需几行代码。那么，为何我们还要花费精力，从零开始去实现它呢？

因为，在亲手构建的过程中，你会遇到意料之外的错误，会为微小的性能提升而反复调试，会为终于跑通一个模块而欣喜。这些“费时”的经历，正是知识内化为直觉的宝贵过程。由此获得的深刻洞察，将成为你驾驭任何高级工具、评估技术方案、甚至进行创新的底气。更重要的是，创造本身，就是最大的乐趣。

现在，一切都已就绪。请打开你的代码编辑器，让我们一同踏上这段“手搓大模型”的激动之旅，亲手揭开智能时代的核心奥秘。

## 目录

### 第一部分 《动手学：机器学习》

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item}
<a href="./notebooks/01机器何以学习.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 1. 机器何以学习
</h4>
<p class="text-truncate">
手动从零构建第一个AI模型，快速掌握机器学习的本质，理解机器是何以学习的。
</p>
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/02神经网络启蒙.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 2. 神经网络启蒙
</h4>
<p class="text-truncate">
使用 PyTorch 训练第一个神经网络模型，快速掌握使用 PyTorch 处理回归任务基本流程。
</p>
</div>
</a>
:::
::::

### 第二部分 《动手学：深度学习》

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item}
<a href="./notebooks/03深度学习初探.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 3. 深度学习初探
</h4>
<p class="text-truncate">
从回归问题升级到多分类问题，快速掌握使用PyTorch处理多分类任务的基本流程。
</p>
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/04PyTorch Lightning 重构.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 4. 工程的进化
</h4>
<p class="text-truncate">
使用 PyTorch Lightning 重构咖啡风味质检模型。
</p>
</div>
</a>
:::
::::

### 第三部分 《动手学：循环神经网络》

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item}
<a href="./notebooks/05语言的序章.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 5. 语言的序章
</h4>
<p class="text-truncate">
迈出自然语言处理的第一步，构建并训练一个用于情感分类的神经网络模型。
</p>
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/06表示的困境.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 6. 表示的困境
</h4>
<p class="text-truncate">
通过将文本从简单的字符索引序列升级为结构化的One-Hot向量，我们成功解决了上一章中因ID数值任意性导致的模型不稳定与过拟合问题。
</p>
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/07记忆的萌芽.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 7. 记忆的萌芽
</h4>
<p class="text-truncate">
系统阐述循环神经网络（RNN）的基本原理与实现方法。通过引入循环结构，RNN能够将历史信息（即隐藏状态）传递至当前时刻，从而有效捕捉序列数据中的时序依赖关系。
</p>
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/08语义的飞跃.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 8. 语义的飞跃
</h4>
<p class="text-truncate">
本章系统性地介绍了词嵌入（Word Embedding）技术，这是自然语言处理中从离散符号表示转向连续语义表示的关键突破。
</p>
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/09记忆的进化.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 9. 记忆的进化
</h4>
<p class="text-truncate">
本章重点介绍了Gated RNN，旨在解决简单RNN中普遍存在的梯度消失或梯度爆炸问题。我们指出，以LSTM和GRU为代表的门控循环神经网络通过引入“门”机制，能够对信息流进行精细调控，选择性地保留关键历史信息并过滤无关内容，从而更有效地控制数据与梯度的传递。
</p>
</div>
</a>
:::
::::

### 第四部分 《动手学：Seq2Seq》

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item}
<a href="./notebooks/10穷举的困境.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 10. 穷举的困境
</h4>
<p class="text-truncate">
通过本次任务，你将会尝试解决一个实际问题：如何让机器“理解”加法运算，并认知到分类模型的局限性。
</p>
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/11从理解到创造.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 11. 从理解到创造
</h4>
<p class="text-truncate">
通过本次任务，你将学会如何使用 Seq2Seq 生成式模型解决加法计算问题。
</p>
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/12技巧的力量.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 12. 改进Seq2Seq模型：技巧的力量
</h4>
<p class="text-truncate">
通过本次任务，你将学会如何使用反转输入改进 Seq2Seq 生成式模型的效果。
</p>
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/13信息的价值.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 13. 改进Seq2Seq模型：信息的价值
</h4>
<p class="text-truncate">
通过本次任务，你将学会如何使用信息偷窥改进 Seq2Seq 生成式模型的效果。
</p>
</div>
</a>
:::
::::

### 第五部分 《动手学：Transformer》

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item}
<a href="./notebooks/14注意力革命.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 14. 注意力革命
</h4>
<p class="text-truncate">
本章将系统介绍 Transformer 模型及其工作原理。
</p>
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/15定义 Transformer 分词器.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 15. 定义 Transformer 分词器
</h4>
<p class="text-truncate">
本章我们将介绍 Transformer 模型的数据需求以及如何定义 Transformer 的分词器。
</p>
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/16构造 Transformer 数据集.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 16. 构造 Transformer 数据集
</h4>
<p class="text-truncate">
本章将学习构造训练 Transformer 模型所需的数据集的方法。
</p>
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/17实现 Transformer Input 组件.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 17. 实现 Transformer Input 组件
</h4>
<p class="text-truncate">
本章将介绍如何实现位置编码等 Transformer 输入模块相关组件。
</p>
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/18实现 Transformer Encoder 组件.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 18. 实现 Transformer Encoder 组件
</h4>
<p class="text-truncate">
介绍如何实现自注意力层、前馈网络层等 Transformer 编码器相关组件。
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/19实现 Transformer Decoder 组件.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 19. 实现 Transformer Decoder 组件
</h4>
<p class="text-truncate">
本章将介绍如何实现 Transformer 模型的解码器组件。
</p>
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/20实现完整的 Transformer 模型.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 20. 实现完整的 Transformer 模型
</h4>
<p class="text-truncate">
本章将串联编码器和解码器，并实现完整的 Transformer 模型。
</p>
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/21Transformer 模型训练和评估.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 21. Transformer 模型训练和评估
</h4>
<p class="text-truncate">
本章将介绍如何训练和评估 Transformer 模型。
</div>
</a>
:::

::::

### 第六部分 《动手学：从零训练 GPT》

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item}
<a href="./notebooks/22从零构建 GPT 模型.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 22. 从零构建 GPT 模型
</h4>
<p class="text-truncate">
本章将介绍如何从零构建 GPT 模型。
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/23定义 GPT 分词器.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 23. 定义 GPT 分词器
</h4>
<p class="text-truncate">
本章将介绍如何定义 GPT 模型所需要的分词器。
</p>
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/24构造 GPT 预训练数据集.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 24. 构造 GPT 预训练数据集
</h4>
<p class="text-truncate">
本章将介绍如何构造 GPT 模型的预训练数据集。
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/25实现 GPT 嵌入层.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 25. 实现 GPT 嵌入层
</h4>
<p class="text-truncate">
本章将介绍如何实现 GPT 模型的嵌入层。
</p>
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/26实现 GPT 层归一化.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 26. 实现 GPT 层归一化
</h4>
<p class="text-truncate">
本章将介绍如何实现 GPT 模型的层归一化。
</p>
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/27实现 GPT 多头注意力层.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 27. 实现 GPT 多头注意力层
</h4>
<p class="text-truncate">
本章将介绍如何实现 GPT 模型的多头注意力层。
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/28实现 GPT 前馈网络层.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 28. 实现 GPT 前馈网络层
</h4>
<p class="text-truncate">
本章将介绍如何实现 GPT 模型的前馈网络层。
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/29定义 Transformer Block 模块.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 29. 定义 Transformer Block 模块
</h4>
<p class="text-truncate">
本章将介绍如何定义 Transformer 模型的 Block 模块。
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/30构建完整的 GPT 模型.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 30. 构建完整的 GPT 模型
</h4>
<p class="text-truncate">
本章将介绍如何构建完整的 GPT 模型。
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/31实现 GPT 模型的生成策略.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 31. 实现 GPT 模型的生成策略
</h4>
<p class="text-truncate">
本章将介绍如何实现 GPT 模型的生成策略。
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/32GPT 模型的训练与评估.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 32. GPT 模型的训练与评估
</h4>
<p class="text-truncate">
本章将介绍如何训练和评估 GPT 模型。
</div>
</a>
:::

:::{grid-item}
<a href="./notebooks/33更具多样性的生成策略.html">

<div class="alert alert-light simple-tilt2 custom-card">
<h4 class="mt-3"> 33. 更具多样性的生成策略
</h4>
<p class="text-truncate">
本章将介绍如何实现 GPT 模型的更具多样性的生成策略。
</div>
</a>
:::

::::

## 版权

本作品由 `吾辈亦有感` 创作，并已完成著作权登记。

为保护创作成果与读者体验，请遵守以下规则：

🚫 请勿复制或镜像全站内容，建立类似站点；

🚫 请勿整理全站内容，制作或分发电子书；

🚫 请勿将本作品用于任何商业用途。

本站文章包含大量图示、代码及自定义样式，旨在为读者提供最佳的阅读与学习体验。如需分享，请直接转发原始文章链接，以便读者获取最新版本并享受完整的阅读设计。感谢您的理解与支持。

## 讨论

```{div} alert alert-light simple-tilt custom-card

<a href="https://t.zsxq.com/mAjcy">
    <div>
        <div style="display: flex; align-items: center">
            <img src="https://fasterai-picgo.oss-cn-beijing.aliyuncs.com/logo.png" style="width: 128px">
            <div style="margin-left: 20px">
                <h4 class="card-title mt-3">答疑讨论</h4>
                <p class="card-text">
                    如果你有任何学习上的疑问，可以评论留言，和我一起讨论。我会抽空回复你的问题，也欢迎你回答其他人的问题。
                </p>
            </div>
        </div>
    </div>
</a>
```

## 致谢

谨向参与本书 Beta 版试读的悟空、磊哥、至尊宝、格物致知、XiangHe、欢乐Ma、稳中向好、飞凡等诸位朋友致以诚挚谢意。你们的指正与建议，助我克服了‘知识诅咒’，提升了内容的可理解性。

## 推荐

在学习人工智能的过程中，经常需要查阅海外的前沿论文、访问 GitHub 上的开源项目、下载最新的 AI 模型，或者试用一些需要“特殊网络环境”的 AI 工具和模型。这时候，一个稳定、快速的网络通道就显得尤为重要。

我自己一直在用 FLYINGBIRD，体验很不错，推荐给同样在 AI 路上探索的朋友们：

[科学上网-邀请链接](https://fbinv02.fbaff.cc/auth/register?code=OK6z3k5W)
