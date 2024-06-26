# Attention is All You Need

**加粗**为不会翻译

# 摘要

主导的序列转录模型基于复杂的循环或卷积神经网络，都包括一个编码器和一个解码器。性能最好的模型还通过一种注意力机制连接编码器和解码器。我们提出了一种新的简单网络结构，Transformer，仅仅基于注意力机制，完全抛弃循环和卷积。在两个机器翻译任务上的实验表明这些模型质量更好，同时更加并行且需要明显更少的训练时间。我们的模型在WMT2014英语-德语翻译任务中达到28.4BLEU，超越现有最好结果包括**ensembles(想对于单一模型，模型组合之类)** 2BLEU以上。在WMT2014英语-法语翻译任务中，在8张GPU上训练了3.5天后，我们的模型取得最好的单模型BLEU分数，41.8，训练成本仅仅是来自**the literature**的最好模型的一小部分。通过成功应用Transformer到英语成分分析，无论有大量还是有限训练数据，表明它可以很好地泛化到其他任务。

# 1 介绍

循环神经网络，长短期记忆，尤其是门控循环神经网络，已经被稳固的确立为序列构造和转录任务的最好方法，例如语言构造和机器翻译。大量努力因此继续推动循环语言模型和编码器-解码器结构的边界。

循环模型通常随着输入和输出序列的符号位置分解计算。**对齐位置和计算时间步数**，它们产生一个隐藏状态$h_t$的序列，作为先前隐藏状态$h_{t-1}$和输入位置$t$的函数。这种固有的顺序性排除了训练的并行性，这在较长的序列长度下变得关键，因为内存约束限制了**batching across examples**。近期工作已经通过**factorization**技巧和状态计算在计算效率上取得显著提升，同时后者还提高了模型性能。然而序列计算的根本限制依然存在。

注意力机制已经在许多任务中成为**compelling**序列构造和转录模型的必要的一部分，使得构造**dependencies**不用考虑在它们输入和输出序列中的距离。然而在大部分例子中，这样的注意力机制通常和循环网络一起被使用。

在这项工作中我们提出了Transformer，一种模型架构抛弃了循环，而是完全依赖于注意力机制来**draw**输入和输入之间的全局**dependencies**。

# 2 背景

减少序列计算量的目标还建立了扩展神经GPU、ByteNet和ConvS2S的基础，它们全用卷积神经网络作为基础结构块，并行计算所有输入和输出位置的隐藏表征。在这些模型中，**关联**来自两个任意输入或输出位置的**信号**需要的操作次数随着位置之间的距离增长，ConvS2S是线性增长的，ByteNet是对数增长的。这导致学习远距离位置的关系更加困难。在Transformer中，这减少到了常数次操作，尽管代价是由于平均注意力权重位置而减少了有效分辨率，我们用多头注意力抵消这种影响，将在3.2节中介绍。

自注意力，有时也叫做内注意力，是一种注意力机制，关联单一序列的不同位置来计算序列表征。自注意力已经被成功用在许多任务上，包括阅读理解、抽象总结、文本内涵和学习**任务无关**句子表征。

端到端的记忆网络基于循环注意力机制而不是对齐序列循环，已经被表明在简单语言问题回答和语言构造任务上表现良好。

然而，据我们所知，Transformer是第一个完全依赖于自注意力而不是序列对齐循环神经网络或卷积来计算输入和输出的表征的转录模型。在接下来的章节中，我们将介绍Transformer，**motivate**自注意力并且讨论它相对于其他模型（例如[17, 18]和[9]）的优点。

# 3 模型架构

大部分有竞争力的神经序列转录模型有一个编码器-解码器结构。在这，编码器将一个符号表征$(x_1,...,x_n)$的输入序列映射到一个连续表征$\mathbf{z} = (z_1,...,z_n)$的序列。给定$\mathbf{z}$，解码器一次生成输出符号序列$(y_1,...,y_n)$的一个元素。模型每一步是自回归的，生成下一个符号时将上一个生成的符号作为附加的输入。

Transformer遵循这个总体架构，在编码器和解码器中都用堆栈的自注意力和***point-wise(写错了?)***全连接层，分别如图1中左右所示。

## 3.1 编码器和解码器堆栈

### 编码器：

编码器由$N=6$个相同层的堆栈组成。每层有两个子层。第一个是多头自注意力机制，第二个是简单的基于位置的全连接前馈网络。我门在每个子层前后利用残差连接，随后是层归一化。也就是每个子层的输出是$LayerNorm(x + Sublayer(x))$，其中$Sublayer(x)$是子层自身实现的函数。为了促进这些残差连接，模型中的所有子层和内嵌层，产生$d_{model}=512$维的输出。

### 解码器：

解码器同样由$N=6$个相同层的堆栈组成。除了每个编码器层的两个子层，解码器插入了第三个子层，在编码器堆栈的输出上使用多头自注意力。与编码器相似，我们在每个子层前后利用残差连接，随后是层归一化。我们还修改了解码器堆栈自注意力子层来防止位置注意后续位置。这个掩蔽，结合输出内嵌偏移一个位置的事实，确保位置$i$的预测仅仅依赖位置在$i$之前的已知输出。

## 3.2 注意力

一个注意力函数可以被描述为将一个查询和一组键值对映射到一个输出，其中查询、键、值和输出都是向量。输出作为值的权重和被计算，其中分配给每个值的权重是通过查询和对应键的兼容性函数计算的。

### 3.2.1 缩放点积注意力

我们称我们特指的注意力为“缩放点积注意力”（图2）。输出由$d_k$维的查询和键以及$d_v$维的值构成。我们计算查询和所有键的点积，每个除以$\sqrt{d_k}$，并应用softmax函数来获得值的权重。

实际上，我们同时在一组查询上计算注意力函数，组合成矩阵$Q$。键和值同样组合成矩阵$K$和$V$。我们计算输出矩阵如下:
$$
Attention(Q,K,V) = softmax(\frac{Q K^T}{\sqrt{d_k}}) V
$$
两种最常用的注意力函数是加性注意力和点积（乘法）注意力。点积注意力与我们的算法相同，除了缩放因子$\frac{1}{\sqrt{d_k}}$。加性注意力用一个单隐藏层的前馈网络计算兼容性函数。尽管二者的理论复杂度相似，但是点积注意力实际上更快且空间效率更高，因此它可以被用高度优化的矩阵乘法代码实现。

尽管$d_k$很小时两种机制表现相似，但是不缩放更大的$d_k$时加性注意力由于点积注意力。我们怀疑$d_k$很大时，点积变得很大，将softmax函数推入梯度极小的区域。为了抵消这个影响，我们缩放点积$\frac{1}{\sqrt{d_k}}$倍。

### 3.2.2 多头注意力

不使用$d_{model}$维的键、值和查询的单个注意力函数，我们发现效果好的是用不同的可学习的线性投影分别将查询、键和值线性投影$h$次到$d_k$、$d_k$和$d_v$维。我们在每个查询、键和值投影后的版本上并行执行注意力函数，输出$d_v$维的输出值。这些输出被连接并再一次投影，结果是最终值，如图2描述。

多头注意力允许模型共同注意来自不同(在不同位置的表征子空间)的信息。对于单头注意力，**averaging ingibits this**。
$$
MultiHead(Q,K,V) = Concat(head_1,...,Head_h) W^O \\
where \; head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
$$
其中投影是参数矩阵$W_i^Q \in \mathbb{R} ^ {d_{model} \times d_k}$，$W_i^K \in \mathbb{R} ^ {d_{model} \times d_k}$，$W_i^V \in \mathbb{R} ^ {d_{model} \times d_v}$和$W^O \in \mathbb{R} ^ {h d_v \times d_{model}}$。

在这项工作中，我们使用$h=8$个平行注意力层，或者头。对于每个头我们使用$d_k = d_v = d_{model} / h = 64$。由于每个头的维度减少，总计算成本和全维度单头注意力相似。

### 3.2.3 我们模型中应用的注意力

Transformer以三种不同的方式用多头注意力：

- 在“编码器-解码器注意力”层中，查询来自前一解码器层，**记忆**键和值来自编码器的输出。这允许解码器的所有位置注意输入序列的所有位置。这像典型的序列-到-序列模型的编码器-解码器注意力机制例如[38, 2, 9]。
- 编码器包含自注意力层。在一个自注意力层中，所有的键、值和查询来自相同的地方，在这种情况下，是编码器中前一层的输出。编码器的每个位置可以注意编码器前一层的所有位置。
- 类似，解码器的自注意力层允许解码器的每个位置注意之前并包括自己的所有位置。我们需要防止解码器的左向信息流来保持自回归性。我们在缩放点积注意力内部通过掩蔽（设置为$-\infty$）所有对应非法连接的softmax输入值实现。见图2。

## 3.3 基于位置的前馈网络

住了注意力子层，在我们的编码器和解码器中的每层包括一个全连接前馈网络，被分别相同的应用于每个位置。这由两个线性变换和中间的ReLU激活组成。
$$
FFN(x) = \max(0, x W_1 + b_1) W_2 + b_2
$$
尽管线性变换在不同位置是相同的，但是每个层它们用不同的参数。另一种描述这个的方式是当作两个核尺寸为1的卷积。输入和输出的维度是$d_{model}=512$，内部层有$d_{ff}=2048$维。

## 3.4 内嵌和Softmax

和其他序列转录模型类似，我们学习内嵌来将输入和输出tokens转换为$d_{model}$维向量。我们还用常见的可学习线性变换和softmax函数来将解码器输出转换到预测下一token的可能性。在我们的模型中，我们在两个内嵌层和softmax前的线性变换层之间共享权重矩阵，类似[30]。在内嵌层，我们将这些权重乘以$\sqrt{d_{model}}$。

## 3.5 位置编码

由于我们的模型不包含循环和卷积，为了模型能够利用序列顺序，我们必须注入有关tokens在序列中相对或绝对的位置信息。为此，我们在编码器和解码器堆栈的底部添加“位置编码”到输入内嵌中。位置编码有和内嵌相同的维度$d_{model}$，因此两者可以求和。有许多可选的位置编码，可学习的或固定的。

我这项工作中，我们使用不同频率的正弦和余弦函数：
$$
PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{model}}) \\
PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{model}})
$$
其中$pos$是位置，$i$是维度。也就是说，位置编码的每个维度和一个正弦曲线有关。几何级数的波长从$2\pi$到$10000 \cdot 2\pi$。我们选择这个函数是因为我们假设它将允许模型容易学会通过相对位置来关注，因此对于任何固定的偏移$k$，$PE_{pos+k}$可以被表示为$PE_{pos}$的一个线性函数。

我们还用可学习的位置编码代替它做实验，发现两个版本产生几乎一样的结果（见表3行E）。我们选择正弦曲线版本是因为它可能允许模型外推长度比训练时遇到的序列更长的序列。

# 4 为什么自注意力

在这一章节中，我们对比自注意力层和循环、卷积层的许多方面，常常用来将一个长度可变的符号表征序列$(x_1,...,x_n)$映射到另一个长度相同的序列$(z_1,...,z_n)$，其中$x_i,z_i \in \mathbb{R}^d$，比如典型序列转录编码器或解码器的隐藏层。**激发**我们使用自注意力，我们考虑了三个需求。

一是每个层的总计算复杂度。二是可并行的计算量，通过所需最少顺序操作数来衡量。

三是网络中长距离依赖的路径长度。学习长距离依赖是许多序列转录任务的关键挑战。一个影响学习这类依赖的能力的关键要素是前向和后向信号在网络中传播的必经路经长度。这些输入和输出序列中任意位置结合之间的路径越短，越容易学习到长距离依赖关系。因此我们还对比了由不同类型层组成的网络中任意两个输入和输出位置的最长路径距离

如表1所示，一个自注意力层用常数级的顺序执行操作连接所有位置，然而一个循环层需要$O(n)$顺序操作。在计算复杂度方面，当序列长度$n$比特征维度$d$小时，自注意力层比循环层更快，这是最好的机器翻译模型用的句子表征中最常见的情况，例如word-piece和byte-pair表征。为了提高涉及非常长序列的任务的计算性能，自注意力可被限制为只注意以对应输出位置为中心的输入序列的尺寸为$r$的上下文。这使最大路径长度增至$O(n/r)$。我们计划在未来的工作中进一步研究这个方法。

单独一个核宽度$k<n$的卷积层不能连接所有输入和输出位置对。这样做需要连续核时$O(n/k)$个卷积层的堆栈，或者空洞卷积时$O(log_k(n))$，增加了网络中任意两个位置之间的最长路径的长度。卷积层通常比循环层贵$k$倍。但是可分离卷积显著减少复杂度到$O(k \cdot n \cdot d + n \cdot d^2)$。但是即使$k=n$时，可分离卷积的复杂度也就和一个自注意力层以及一个基于位置的前馈网络的结合相同，后者即我们在我们的模型中采用的方法。

副作用是自注意力会产生更多可解释模型。我们查看我们模型中的注意力分布并在附录中给出和讨论例子。不仅单注意力头清楚的学会了执行不同的任务，多头似乎表现出有关语法和语义结构有关的行为。

# 5 训练

该章节描述了我们模型的训练方法。

## 5.1 训练数据和批处理

我们在包含大约450万句子对的标准WMT2014英语-德语数据集上训练。句子用 byte-pair encoding编码为大约有37000tokens的来源-目标共享词表。对于英语-法语，我们用明显更大的WMT2014英语-法语数据集，包含3600万个句子，分token成32000word-piece词表。句子对按照大致序列长度打包在一起。每个训练批包含一组包含大约25000源token和25000目标token的句子对。

## 5.2 硬件和安排

我们在一台有8张英伟达P100显卡的机器上训练我们的模型。基础模型用了在文中描述的超参数，每一训练步化了大约0.4秒。我们训练基础模型总共100,000步或12小时。对于大模型，（在表3最下行描述），步时间是1.0秒。大模型训练300,000步（3.5天）。

## 5.3 优化器

我们使用Adam优化器，其中$\beta_1=0.9$​，$\beta_2=0.98$​和$\epsilon=10^{-9}$​。我们根据以下公式改变整个训练过程中的学习率：
$$
lrate = d_{model}^{-0.5} \cdot \min(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5})
$$
这对应在前$warmup\_steps$个训练步是线性增加学习率，并且此后与步数的平方根成比例的减少学习率。我们用$warmp\_steps=4000$。

## 5.4 正则化

我们在训练中用三种正则化：

### 残差dropout

我们对每个子层的输出应用dropout，在它被加上子层输入并归一化前。此外。我们对编码器和解码器堆栈的内嵌和位置编码的和应用dropout。对于基础模型，我们用比率$P_{drop}=0.1$。

### 标签光滑

在训练中，我们用值为$\epsilon_{ls}=0.1$的标签光滑。这有害困惑度，因为模型学习更加不确定，但是提高了准确度和BLEU分数。

# 6 结果

## 6.1 机器翻译

在WMT2014英语-德语翻译任务，大transformer模型（表2的Transformer(big)）超过先前最好的报告了的模型（包括**ensembles**)2.0BLEU以上，达到了最好的BLEU分数，28.4。这个模型的配置在表3最底部行列出。在8张P100显卡上训练消耗3.5天。即使我们的基础模型也超越了所有先前公开的模型和**ensembles**，训练成本仅为任何有竞争力的模型的一小部分。

在WMT2014英语-法语翻译任务上，我们的大模型取得了41.0BLEU分数，超越了所有先前公开的单一模型，训练成本仅为先前最好模型的1/4不到。Transformer(big)模型在英语-法语训练用dropout比率$P_{dropout}=0.1$，而不是0.3。

对于基础模型，我们使用平均最后5个写入间隔为10分钟的检查点得到的单一模型。对于大模型，我们平均最后20个检查点。我们用集束搜索，集束尺寸为4，长度惩罚$\alpha=0.6$。超参数在实验开发集后选择。我们设置推理最大输出长度为输入长度+50，但是有可能时更早结束。

表2总结了我们的结果并比较我们相对于其他来自**litearture**的模型架构的翻译质量和训练成本。我们通过乘训练时间，使用的GPU数量和每个GPU的持续单精度浮点容量的估计，估计用来训练一个模型的浮点操作数。

## 6.2 模型变体

为了评估Transformer不同组件的重要性，我们用不同方式改变我们的基础模型，测量改变在英语-德语翻译开发集newstest2013上的表现。我们使用在先前章节提到过的集束搜索，但是没有平均检查点。我们在表3中展示结果。

在表3行A中，我们改变注意力头数和注意力键和值维度，保持3.2.2节中提到的计算常数量。虽然单头注意力比最好的设置差了0.9BLEU，但是头数太多时质量同样下降。

在表3行B中，我们发现减少追忆里键尺寸$d_k$有害模型质量。这表示确定兼容性并不容易，比点积更先进的兼容性函数可能有益。我们进一步观察行C和D，如期望的，更大的模型更好，dropout对于防止过拟合十分有用。在行E中我们用可学习位置编码代替我们的正弦曲线位置编码，发现结果和基础模型几乎一样。

## 6.3 英语成分分析

为了评估Transformer是否能够泛化到其他任务，我们在英语成分分析上进行实验。这个任务提出了具体的挑战：输出受到强烈的结构约束并且明显长于输出。此外，RNN序列到序列模型还不能够在小数据方法上取得最好的结果。

我们在Penn Treebank的部分WSJ的4万训练句子上，训练了一个4层transformer，其中$d_{model}=1024$。我们还在半监督**环境**下训练它，用更大的**高度信心**和BerkleyParser预料，来自大约1700万个句子。我们为仅WSJ**设置**用1万6千个token的词表，为半监督**环境**用3万2千个token的词表。

我们仅在**22节开发集**上做了少量实验来选择dropout，注意力和残差（5.4节），学习率和集束尺寸，所有其他超参数按照英语-德语基础翻译模型保持不变。在推理中，我们增加最大输出长度到输入长度+300。我们对**仅**WSJ和半监督**环境**都用集束尺寸为21，$\alpha=0.3$。

表4中我们的结果表明，尽管缺少特定任务的调整，我们的模型表现的出奇的好，产生的结果比除了卷积神经网络Grammar之外其他所有先前报告的模型都好。

不同于RNN序列到序列模型，Transformer优于Berkeley-Parser，即使只在4万个句子的WSJ训练集上训练。

# 7 结论

在此项工作中，我们提出了Transformer，首个完全基于注意力的序列转录模型，用多头自注意力取代了最常用于编码器-解码器架构的循环层。

对于翻译任务，Transformer可以训练地比基于循环或卷积层的架构更快。在WMT2014英语到德语和WMT2014英语到法语翻译任务上，我们都取得了最好。在前一任务中，我们最好的模型甚至优于所有的先前报告的**ensembles**。

我们对基于注意力的模型的未来充满兴趣，计划应用它们到其他任务。我们计划将Transformer扩展到涉及文本以外的输入和输出模态的问题，并研究局部、受限注意力机制，以有效处理图像、音频和视频等大型输入和输出。减少生成顺序性是我们的另一个研究目标。

我们用来训练和评估的代码可在https://github.com/tensorflow/tensor2tensor获取。