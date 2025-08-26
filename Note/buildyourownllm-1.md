

# 从零开始200行python代码实现LLM


原文章：

https://mp.weixin.qq.com/s/vq_kjizpzr9clvrAf9dkLQ?poc_token=HHrIN2ij3TNUmK1XAYzEf-CFvg3f65zsVCrGc7tK


好的，作为你的高级软件工程师，我将按照你提供的文档，从零开始，一步一步地详细指导你实践所有内容，并极尽详细地解读所有代码、类、方法、变量的原理和作用，确保你彻底理解每一行代码。

我们将严格遵循“finish step by step”的原则。

---

## 从零开始实现LLM：前言与基础Bigram模型

### 1. 目标与适用范围

首先，让我们明确本文档的目标和适用范围：

*   **目标**：从零开始，使用 Python 实现一个极简但完整的大语言模型，通过实际代码具象化各种概念，让你亲手写出 `self-attention` 机制和 `transformer` 模型。
*   **适用范围**：
    *   你只需要具备基本的 Python 编码能力。
    *   我们将实现一个完整的语言模型（尽管由于层数和数据集限制，它目前只能“写诗词”）。
    *   我们不会深入探讨数学和机器学习的原理性知识，只关注“能用”的层面。
    *   我们尽量不依赖抽象层次高的框架，即使使用也会进行解释。
*   **依赖**：唯一需要安装的依赖是 `torch`。在你的 PowerShell 命令行中运行 `pip install torch` 即可。
*   **建议**：强烈建议使用 `VS Code + Jupyter Notebook (.ipynb)` 的组合来调试代码，这将极大帮助你理解每一步的运行和变量变化。

本文将首先介绍“从零基础到 Bigram 模型”，为后续的 LLM 实现打下基础。

### 2. 传统方式实现“诗词生成器”

在深入机器学习之前，我们先用传统的编程思路来实现一个简单的“诗词生成器”，这有助于我们理解语言模型的核心思想。

**数据集观察**：
我们的数据集是 `ci.txt`，其中包含宋和南唐的词。我们的目标是生成类似这些词的内容。

```powershell
# 在你的PowerShell中，如果ci.txt在当前目录下
# 查看文件前8行内容
Get-Content ci.txt -Head 8
```

输出示例：
```
虞美人 李煜春花秋月何时了，往事知多少？小楼昨夜又东风，故国不堪回首月明中。雕栏玉砌应犹在，只是朱颜改。问君能有几多愁？恰似一江春水向东流。
乌夜啼 李煜昨夜风兼雨，帘帏飒飒秋声。
```

**基本思路**：
词是由一系列汉字组成的。一个最简单的想法是：我们可以统计每个汉字后面出现其他汉字的概率。然后，根据这些概率，不断地递归生成“下一个字”，直到生成足够的字数，再截断一部分，就得到了一首“词”。

**具体步骤**：
1.  **准备词汇表**：将 `ci.txt` 中出现的所有不重复的汉字提取出来，形成我们的“词汇表”。词汇表的长度就是 `vocab_size`。
2.  **统计频率**：创建一个 `vocab_size * vocab_size` 大小的二维结构（可以是一个列表的列表或字典），用来统计每个字后面出现其他字的频率。例如，`transition[char_A_id][char_B_id]` 表示字 A 后面出现字 B 的次数。
3.  **计算概率并生成新字**：
    *   根据统计的频率计算概率（将频率除以该字后面所有字的总频率）。
    *   根据这些概率进行随机采样，选择下一个字。
    *   重复此过程，直到生成所需的长度。

#### 2.1 `simplemodel.py` 代码解读

现在，让我们来详细解读 `simplemodel.py` 的代码。

```python
# simplemodel.py
import random # 导入random模块，用于生成随机数

random.seed(42) # 设置随机数种子，确保每次运行结果一致，便于调试和复现。如果注释掉此行，每次运行将得到不同的随机结果。

prompt = "春江" # 定义初始的生成文本，我们将从“春江”开始生成后续的词语。
max_new_token = 100 # 定义模型将生成的新词语的最大数量。

# 打开并读取ci.txt文件。
# 'r' 表示读取模式，encoding='utf-8' 指定文件编码为UTF-8，以正确处理中文字符。
with open('ci.txt', 'r', encoding='utf-8') as f:
    text = f.read() # 将整个文件内容读取到一个字符串变量text中。

chars = sorted(list(set(text))) # 从text中提取所有不重复的字符，并进行排序，形成我们的词汇表（chars）。
vocab_size = len(chars) # 计算词汇表的大小，即不重复字符的数量。

# 创建字符到整数的映射 (string to integer)。
# stoi是一个字典，键是字符，值是该字符在词汇表中的索引（ID）。
stoi = { ch:i for i,ch in enumerate(chars) }
# 创建整数到字符的映射 (integer to string)。
# itos是一个字典，键是字符的索引（ID），值是对应的字符。
itos = { i:ch for i,ch in enumerate(chars) }

# 定义一个编码函数，将字符串转换为整数ID列表。
# 例如，encode("春江") -> [stoi['春'], stoi['江']]
encode = lambda s: [stoi[c] for c in s]
# 定义一个解码函数，将整数ID列表转换为字符串。
# 例如，decode([stoi['春'], stoi['江']]) -> "春江"
decode = lambda l: ''.join([itos[i] for i in l])

# 初始化一个二维列表（矩阵），用于存储字符之间的转换频率。
# transition[i][j] 将表示在字符i后面出现字符j的次数。
# 矩阵的大小是 vocab_size * vocab_size，所有初始值为0。
transition = [[0 for _ in range(vocab_size)] for _ in range(vocab_size)]

# 遍历文本，统计字符间的转换频率。
# 从文本的第一个字符到倒数第二个字符，因为我们需要看当前字符和下一个字符。
for i in range(len(text) - 1):
    current_token_id = encode(text[i])[0] # 获取当前字符的ID。由于encode返回列表，我们取第一个元素。
    next_token_id = encode(text[i + 1])[0] # 获取下一个字符的ID。
    transition[current_token_id][next_token_id] += 1 # 在转换矩阵中，将当前字符ID到下一个字符ID的频率加1。

# 初始化生成序列，从prompt开始。
generated_token = encode(prompt)

# 开始生成新的词语。循环 max_new_token - 1 次，因为prompt已经有一个词了，我们还需要生成 max_new_token - 1 个。
for i in range(max_new_token - 1):
    current_token_id = generated_token[-1] # 获取当前已生成序列中的最后一个字符的ID。
    logits = transition[current_token_id] # 从转换矩阵中获取当前字符后面所有字符的出现频率（得分）。
    total = sum(logits) # 计算所有频率的总和，用于后续的归一化。
    
    # 归一化处理：将频率转换为概率。
    # 如果total为0（即当前字符从未在训练数据中出现过），则所有概率都为0，这将导致random.choices出错。
    # 这里是一个简单的处理，如果total为0，logits将全是0，random.choices会报错。
    # 实际应用中需要更健壮的处理，例如添加平滑项或确保total不为0。
    logits = [logit / total for logit in logits]

    # 根据计算出的概率分布随机选择下一个字符的ID。
    # range(vocab_size) 是所有可能的字符ID。
    # weights=logits 提供每个字符ID被选择的概率。
    # k=1 表示只选择一个字符。
    next_token_id = random.choices(range(vocab_size), weights=logits, k=1)[0]
    generated_token.append(next_token_id) # 将选择的下一个字符ID添加到生成序列中。
    current_token_id = next_token_id # 更新 current_token_id 为新生成的字符ID，为下一次循环做准备。

print(decode(generated_token)) # 将最终生成的整数ID序列解码为可读的字符串并打印。
```

**运行与输出**：
在 PowerShell 中运行：

```powershell
python simplemodel.py
```

你可能会得到类似这样的输出：

```
春江月 张先生疑被。
倦旅。清歌声月边、莼鲈清唱，尽一卮酒红蕖花月，彩笼里繁蕊珠玑。只今古。浣溪月上宾鸿相照。乞团，烟渚澜翻覆古1半吐，还在蓬瀛烟沼。木兰花露弓刀，更任东南楼缥缈。黄柳，
```

**解读输出**：
这个输出看起来像是一首词，但实际上，我们的模型只是一个“下一个词预测器”。它只知道“春”后面大概率是“江”，“江”后面大概率是“月”，以此类推。它并没有真正理解“词”的结构、意义，甚至不知道什么是开头和结尾。这种“意义”是由我们人类读者赋予的。

**与LLM概念的联系**：
*   **词汇表 (Vocabulary) & Tokenizer (分词器)**：
    *   在 `simplemodel.py` 中，`chars` 列表就是我们的词汇表，`vocab_size` 是词汇表大小。
    *   `stoi` 和 `itos` 字典以及 `encode` 和 `decode` 函数共同构成了我们的简易“分词器”。它将原始文本（字符串）转换为模型可以处理的数字序列（token ID），反之亦然。
    *   真实的 LLM 分词器（如 Qwen2.5 的 tokenizer）更复杂，具有更大的词汇表（151643个词汇），并且能够将常用词组（如“阿里巴巴”、“人工智能”）编码成单个 token，这大大提高了编码效率和模型理解能力。而我们的简易分词器中，一个字符永远对应一个 token。
*   **模型 (Model)**：
    *   我们实现的“模型”本质上是自然语言处理中的 **N-gram 模型**，具体来说是 **Bigram 模型**。它假设每个词的出现只依赖于它前面的一个词。
    *   `transition` 矩阵就是我们 Bigram 模型的参数，它存储了所有字符对之间的转换频率。
*   **训练 (Training)**：
    *   代码中遍历 `ci.txt`，统计 `transition` 矩阵的过程，就是我们模型的“训练”过程。我们通过观察大量数据来学习字符之间的共现规律。
*   **推理 (Inference) / 生成 (Generation)**：
    *   根据 `prompt`，循环预测下一个字符，直到达到 `max_new_token` 的过程，就是模型的“推理”或“生成”过程。
    *   推理步骤：
        1.  从 `transition` 矩阵中获取当前字符后面所有字符的“得分”（`logits`）。在机器学习中，`logits` 是模型最后一层的原始输出值，可以理解为每个候选词的“未归一化分数”。
        2.  将 `logits` 进行归一化处理，转换为概率分布。这里我们使用了最简单的线性归一化。
        3.  根据概率分布随机采样，选择下一个字符的 ID。
        4.  将新选择的字符 ID 添加到已生成序列中，并重复此过程。

### 3. 重构：更具“机器学习风格”的Bigram模型

为了更好地理解后续真实的 PyTorch 代码，我们将对 `simplemodel.py` 进行重构，使其更符合机器学习的编程范式。有 PyTorch 背景的同学可以快速浏览本节。

**主要变化**：
1.  **类封装**：将 `Tokenizer` 和 `BigramLanguageModel` 封装成类。
2.  **批处理**：引入 `batch_size` 和 `block_size` 概念，实现批处理数据加载和模型推理，为 GPU 并行计算做准备。
3.  **`forward` 和 `generate` 方法**：模仿深度学习框架中模型的 `forward` 方法（用于计算输出）和 `generate` 方法（用于序列生成）。

#### 3.1 `simplebigrammodel.py` 代码解读

现在，让我们详细解读 `simplebigrammodel.py` 的代码。

```python
# simplebigrammodel.py
import random # 导入random模块，用于生成随机数和进行随机采样
from typing import List # 从typing模块导入List，用于类型注解，提高代码可读性和可维护性

random.seed(42) # 设置随机数种子，确保每次运行结果一致，便于调试和复现。

# 定义全局参数
prompts = ["春江", "往事"] # 定义多个初始的生成文本，我们将从这些prompt开始生成。
max_new_token = 100 # 定义模型将生成的新词语的最大数量。
max_iters = 8000 # 定义“训练”的最大迭代次数。这里并不是真正的训练，只是多次统计频率，但模拟了训练循环。
batch_size = 32 # 定义每个批次（batch）的大小，即每次处理多少个独立的序列。
block_size = 8 # 定义每个序列的最大长度（也称为上下文长度或块大小），即模型在做预测时会考虑前面多少个字符。

# 打开并读取ci.txt文件，与之前相同。
with open('ci.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenizer 类：封装词汇表的编码和解码逻辑
class Tokenizer:
    def __init__(self, text: str):
        # 构造函数，初始化Tokenizer时需要传入整个文本数据
        self.chars = sorted(list(set(text))) # 从文本中提取所有不重复的字符，并排序，作为词汇表。
        self.vocab_size = len(self.chars) # 计算词汇表的大小。
        self.stoi = {ch: i for i, ch in enumerate(self.chars)} # 字符到ID的映射字典。
        self.itos = {i: ch for i, ch in enumerate(self.chars)} # ID到字符的映射字典。

    def encode(self, s: str) -> List[int]:
        # 将字符串编码为整数ID列表。
        return [self.stoi[c] for c in s]

    def decode(self, l: List[int]) -> str:
        # 将整数ID列表解码为字符串。
        return ''.join([self.itos[i] for i in l])

# BigramLanguageModel 类：模拟机器学习模型结构
class BigramLanguageModel():
    def __init__(self, vocab_size: int):
        # 构造函数，初始化模型时需要传入词汇表大小。
        self.vocab_size = vocab_size
        # 初始化转换矩阵。
        # transition[i][j] 表示字符i后面出现字符j的频率。
        # 这是一个 vocab_size * vocab_size 的二维列表，所有初始值为0。
        self.transition = [[0 for _ in range(vocab_size)] for _ in range(vocab_size)]

    def __call__(self, x):
        # 这是一个Python的特殊方法，使得类的实例可以直接像函数一样被调用，例如 model(x)。
        # 它会内部调用forward方法，模仿PyTorch中nn.Module的行为。
        return self.forward(x)

    def forward(self, idx: List[List[int]]) -> List[List[List[float]]]:
        '''
        模型的“前向传播”方法。
        输入idx：一个二维数组，形状为 (B, T)。
            B (Batch Size) 代表批次大小，即同时推理的序列数量。
            T (Sequence Length) 代表每个序列的长度。
            例如：[[1, 2, 3], [4, 5, 6]] 表示同时有2个序列，每个序列长度为3。
        输出logits：一个三维数组，形状为 (B, T, vocab_size)。
            它表示在每个批次中的每个token位置，预测下一个token是词汇表中各个token的“得分”（频率）。
            例如：[[[0.1, 0.2, ...], [0.4, 0.5, ...], ...], ...]
        '''
        B = len(idx) # 获取批次大小。
        T = len(idx[0]) # 获取每个序列的长度。

        # 初始化一个三维列表来存储logits。
        # logits[b][t] 将是一个长度为 vocab_size 的列表，表示在第b个批次的第t个token后面，
        # 词汇表中每个token的出现频率。
        logits = [
            [[0.0 for _ in range(self.vocab_size)] # 内部列表，表示一个token后面所有可能token的频率
             for _ in range(T)] # 中间列表，表示一个序列中所有token的频率
            for _ in range(B) # 外部列表，表示所有批次的频率
        ]

        # 遍历批次和序列中的每个token，计算其下一个token的频率。
        for b in range(B): # 遍历每个批次
            for t in range(T): # 遍历当前批次中的每个token
                current_token = idx[b][t] # 获取当前token的ID。
                # 从预先统计的transition矩阵中获取当前token后面所有可能token的频率。
                # 这一行是Bigram模型的核心：下一个token的预测只依赖于当前token。
                logits[b][t] = self.transition[current_token]

        return logits # 返回计算出的logits。

    def generate(self, idx: List[List[int]], max_new_tokens: int) -> List[List[int]]:
        '''
        序列生成方法。
        输入idx：一个二维数组，形状为 (B, T)，表示初始的prompt序列批次。
        max_new_tokens：需要生成的新token的最大数量。
        输出：一个二维数组，形状为 (B, T + max_new_tokens)，包含原始prompt和生成的新token。
        '''
        for _ in range(max_new_tokens): # 循环生成 max_new_tokens 个新token。
            # 调用模型的forward方法，获取当前序列批次的logits。
            # logits_batch 的形状是 (B, T_current, vocab_size)，其中 T_current 是当前序列的长度。
            logits_batch = self(idx)

            # 遍历每个批次中的序列，进行采样和扩展。
            for batch_idx, logits_per_sequence in enumerate(logits_batch):
                # 在Bigram模型中，我们只需要最后一个token的下一个token的概率来进行预测。
                # logits_per_sequence 是当前批次中一个序列的logits，形状是 (T_current, vocab_size)。
                # logits_per_sequence[-1] 提取了该序列中最后一个token的logits，形状是 (vocab_size)。
                logits_of_last_token = logits_per_sequence[-1]

                # 计算总频率，用于归一化。使用max(sum(logits), 1)防止除以零的情况。
                total = max(sum(logits_of_last_token), 1)
                # 归一化：将频率转换为概率。
                probs_of_last_token = [logit / total for logit in logits_of_last_token]

                # 根据概率分布随机采样下一个token的ID。
                next_token = random.choices(
                    range(self.vocab_size), # 所有可能的token ID
                    weights=probs_of_last_token, # 对应的概率分布
                    k=1 # 采样一个token
                )[0] # random.choices返回一个列表，我们取第一个元素

                # 将新生成的token添加到当前批次的序列中。
                idx[batch_idx].append(next_token)
        return idx # 返回包含生成token的完整序列批次。

# 定义一个辅助函数，用于从整个数据集中随机获取一批数据
def get_batch(tokens: List[int], batch_size: int, block_size: int) -> tuple[List[List[int]], List[List[int]]]:
    '''
    随机获取一批数据x和y用于“训练”。
    x和y都是二维数组，可以用于并行处理。
    其中y数组内的每一个值，都是x数组内对应位置的值的下一个值。
    这种输入-输出对的设计是机器学习监督学习的常见模式。
    
    例如：
    如果原始tokens是 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    选择一个随机起始索引 i = 0
    x = [1, 2, 3] (block_size=3)
    y = [2, 3, 4]
    
    选择另一个随机起始索引 j = 7
    x = [8, 9, 10]
    y = [9, 10, 11] (这里假设原始tokens足够长)
    '''
    # 随机选择 batch_size 个起始索引。
    # range(len(tokens) - block_size) 确保选取的起始索引有足够的后续token来构成一个完整的 block_size 序列。
    ix = random.choices(range(len(tokens) - block_size), k=batch_size)
    
    x, y = [], [] # 初始化输入序列列表x和目标序列列表y
    for i in ix: # 遍历每个随机选取的起始索引
        x.append(tokens[i:i+block_size]) # 从原始token序列中截取一个长度为 block_size 的子序列作为输入x
        y.append(tokens[i+1:i+block_size+1]) # 截取x序列的下一个token序列作为目标y
    return x, y # 返回输入批次x和目标批次y

# --- 主程序执行部分 ---

tokenizer = Tokenizer(text) # 实例化Tokenizer
vocab_size = tokenizer.vocab_size # 获取词汇表大小
tokens = tokenizer.encode(text) # 将整个文本编码为整数ID序列

model = BigramLanguageModel(vocab_size) # 实例化BigramLanguageModel

# “训练”过程：统计transition矩阵
print("开始“训练”（统计频率）...")
start_time = time.time() # 记录开始时间
for iter in range(max_iters): # 循环 max_iters 次来模拟训练迭代
    # 每次迭代随机获取一批数据x和y。
    # x是输入序列批次，y是对应的目标序列批次（即x中每个token的下一个token）。
    x_batch, y_batch = get_batch(tokens, batch_size, block_size)
    
    # 遍历批次中的每个序列和序列中的每个token，更新transition矩阵。
    # 注意：这里我们没有使用模型的forward方法，而是直接修改了模型的内部参数(transition)。
    # 这模拟了最简单的“学习”过程：观察数据并更新统计信息。
    for i in range(len(x_batch)): # 遍历批次中的每个序列
        for j in range(len(x_batch[i])): # 遍历序列中的每个token
            current_token_id = x_batch[i][j] # 获取当前输入token的ID
            next_token_id = y_batch[i][j] # 获取对应的目标token的ID
            model.transition[current_token_id][next_token_id] += 1 # 统计频率

if max_iters > 0:
    elapsed_time = time.time() - start_time
    print(f"“训练”完成，耗时: {elapsed_time:.2f} 秒")

# 推理过程：生成文本
print("\n开始推理（生成文本）...")
# 将初始prompt编码为整数ID列表，并转换为批次形式 (List[List[int]])
prompt_tokens = [tokenizer.encode(prompt) for prompt in prompts]
# 调用模型的generate方法，生成新的token序列
result = model.generate(prompt_tokens, max_new_token)

# 解码并打印生成的结果
print("\n生成结果：")
for tokens_list in result:
    print(tokenizer.decode(tokens_list)) # 将生成的token ID列表解码为字符串
    print('-'*10) # 分隔符
```

**运行与输出**：
在 PowerShell 中运行：

```powershell
python simplebigrammodel.py
```

你可能会得到类似这样的输出：

```
开始“训练”（统计频率）...
“训练”完成，耗时: 约几秒钟

开始推理（生成文本）...

生成结果：
春江红紫霄效颦。
怎。兰修月。两个事对西风酒伴寄我登临，看雪惊起步，总不与泪满南园春来。最关上阅。信断，名姝，夜正坐认旧武仙 朱弦。
岁，回。

看一丝竹。愿皇受风，当。
妆一笑时，不堪----------
往事多闲田舍、十三楚珪酒困不须紫芝兰花痕皱，青步虹。暗殿人物华高层轩者，临江渌池塘。三峡。天、彩霞冠燕翻云垂杨、一声羌笛罢瑶觥船窗幽园春生阵。长桥。无恙，中有心期。
开处。燕姹绿遍，烂□----------
```

**解读重构后的代码**：

1.  **机器学习风格的一些约定**：
    *   **`Tokenizer` 类**：
        *   将词汇表的构建 (`chars`, `vocab_size`) 和编码/解码逻辑 (`stoi`, `itos`, `encode`, `decode`) 封装在一个类中。这使得分词器成为一个独立的、可复用的组件，与模型分离。
        *   `__init__(self, text: str)`：构造函数，接收完整文本 `text` 来构建词汇表。
        *   `encode(self, s: str) -> List[int]`：将输入字符串 `s` 转换为一个整数 ID 列表。
        *   `decode(self, l: List[int]) -> str`：将输入的整数 ID 列表 `l` 转换回字符串。
    *   **`BigramLanguageModel` 类**：
        *   模仿 PyTorch 中 `nn.Module` 的写法，将模型逻辑封装在一个类中。
        *   `__init__(self, vocab_size: int)`：构造函数，初始化模型参数。这里唯一的参数就是 `transition` 矩阵。
        *   `__call__(self, x)`：Python 特殊方法，使得 `model(x)` 这种调用方式成为可能，它会内部调用 `forward` 方法。这是 PyTorch 模型常用的接口。
        *   `forward(self, idx: List[List[int]]) -> List[List[List[float]]]`：
            *   这是模型进行核心计算（“前向传播”）的地方。它接收一个批次的输入 `idx`（形状为 `(B, T)`，即 `batch_size` 乘以 `block_size`），并计算出每个位置的下一个 token 的 `logits`（分数）。
            *   **`B` (Batch Size)**：批次大小，表示同时处理的独立序列的数量。例如，如果你有 32 个 `prompt`，那么 `B` 就是 32。
            *   **`T` (Time Steps or Sequence Length)**：序列长度，表示每个序列包含的 token 数量。在我们的 `get_batch` 中，它对应 `block_size`。
            *   **`C` (Channels or Feature Dimension)**：在当前 Bigram 模型中，`C` 对应 `vocab_size`，因为我们最终要预测的是词汇表中每个词的概率。
            *   `logits` 的输出形状为 `(B, T, vocab_size)`。这意味着对于批次中的每个序列 (`B` 个)，序列中的每个 token (`T` 个)，都会预测一个长度为 `vocab_size` 的分数列表，表示下一个 token 是哪个词汇表中词的可能性。
            *   尽管 Bigram 模型实际上只需要序列中最后一个 token 的 `logits` 来预测下一个词，但为了与后续更复杂的模型保持一致，`forward` 方法仍然为序列中的每一个 `T` 位置都计算了 `logits`。
        *   `generate(self, idx: List[List[int]], max_new_tokens: int) -> List[List[int]]`：
            *   这是模型根据给定的初始序列 `idx`（`prompt`）生成新序列的方法。
            *   它在一个循环中重复调用 `forward` 方法，获取下一个 token 的 `logits`，然后进行采样，并将新生成的 token 添加到序列末尾。
            *   这里使用了 `logits = logits_per_sequence[-1]` 来提取每个序列中最后一个 token 的预测结果，因为 Bigram 模型只依赖于前一个 token。
            *   `total = max(sum(logits_of_last_token), 1)`：在计算概率时，为了避免 `total` 为 0 导致除零错误，这里添加了 `max(..., 1)` 的处理。
            *   `random.choices(...)`：根据概率分布随机选择下一个 token。
    *   **数据加载机制 (`get_batch` 函数)**：
        *   `get_batch(tokens, batch_size, block_size)`：这个函数负责从整个文本数据中随机抽取一批数据，用于模拟训练。
        *   `ix = random.choices(range(len(tokens) - block_size), k=batch_size)`：随机选择 `batch_size` 个起始索引 `i`。`len(tokens) - block_size` 确保我们总能截取到完整的 `block_size` 长度的序列。
        *   `x.append(tokens[i:i+block_size])`：`x` 是模型的输入，它是一个长度为 `block_size` 的 token 序列。
        *   `y.append(tokens[i+1:i+block_size+1])`：`y` 是模型的目标输出，它对应 `x` 中每个 token 的下一个 token。
        *   这种 `(x, y)` 的输入-目标对是监督学习中的标准格式，`x` 是输入特征，`y` 是对应的标签。

2.  **批处理 (Batching) 与张量 (Tensor) 的概念**：
    *   这个版本最显著的变化是数据都以多维数组（Python 中的 `List[List[int]]`）的形式进行输入和输出。在机器学习中，这些多维数组被称为 **张量 (Tensor)**。
    *   **张量** 是数学和物理学中用于表示多维数据的对象，在深度学习框架（如 TensorFlow 和 PyTorch）中，它是数据的基本结构。
    *   引入批处理的目的是为了 **高效利用硬件（尤其是 GPU）进行并行计算**。虽然我们目前的 Python `for` 循环仍然是串行的，但这个数据结构为未来的并行化奠定了基础。
    *   **`forward` 函数中的张量形状**：
        *   输入 `idx` 的形状是 `(B, T)`。
        *   输出 `logits` 的形状是 `(B, T, C)`，其中 `C` 就是 `vocab_size`。
        *   理解 `logits`：想象成 `B * T` 个独立的预测任务。每个任务都是预测一个 token 的下一个 token。因此，每个 `(b, t)` 位置都会有一个长度为 `vocab_size` 的向量，表示下一个 token 是词汇表中哪个词的“得分”。
        *   在 Bigram 模型中，`logits[b][t]` 实际上只依赖于 `idx[b][t]`，而与 `idx[b][0]` 到 `idx[b][t-1]` 无关。但这种 `(B, T, C)` 的输出结构是通用的，为后续更复杂的模型（如 Transformer）做好了准备，因为那些模型确实需要考虑整个序列的上下文。

**总结**：
现在我们有了一个能运行的“玩具模型”，它能够根据统计概率预测下一个词，并且代码结构已经初步具备了机器学习模型的特征。但是，它仍然缺乏真正的“训练”过程，即通过梯度下降等优化算法来调整模型参数。为了实现真正的机器学习，我们将引入 PyTorch。

---

### 4. 5分钟简明PyTorch教程

PyTorch 是一个开源的深度学习库，它提供了方便的数据结构（张量）和函数，极大地简化了深度学习模型的开发。

#### 4.1 `pytorch_5min.py` 代码解读

我们将通过一个简单的线性回归例子来快速了解 PyTorch 的基本用法。

```python
# pytorch_5min.py
import torch # 导入PyTorch库
from torch import nn # 从torch导入神经网络模块，nn包含了各种神经网络层（如线性层、卷积层等）
from torch.nn import functional as F # 导入神经网络函数模块，F包含了激活函数、损失函数等无状态操作

torch.manual_seed(42) # 设置PyTorch的随机数种子，确保每次运行结果一致。

# 判断环境中是否有GPU（CUDA）或MPS（Apple Silicon GPU）可用，并选择最快的设备。
# 'cuda' 用于NVIDIA GPU
# 'mps' 用于Apple Silicon Mac的Metal Performance Shaders
# 'cpu' 用于CPU
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
print(f"Using {device} device") # 打印当前使用的设备

# 1. 创建tensor演示
# torch.tensor() 用于创建张量。张量是PyTorch中数据的基本单位，类似于NumPy数组，但可以在GPU上加速。
x = torch.tensor([1.0, 2.0, 3.0]) # 创建一个一维张量x
y = torch.tensor([2.0, 4.0, 6.0]) # 创建一个一维张量y

# 2. 基本运算演示
print(x + y) # 张量逐元素加法: tensor([3., 6., 9.])
print(x * y) # 张量逐元素乘法 (点乘): tensor([2., 8., 18.])
print(torch.matmul(x, y)) # 矩阵乘法（对于一维向量，这相当于点积）: tensor(28.) (1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28)
print(x @ y) # 另一种矩阵乘法的写法（Python 3.5+ 支持的运算符）: tensor(28.)
print(x.shape) # 获取张量的形状 (维度信息): torch.Size([3])

# 3. 定义模型：一个简单的线性网络
# 在PyTorch中，模型通常继承自nn.Module。
class SimpleNet(nn.Module):
    def __init__(self):
        # 构造函数，初始化模型层。
        super().__init__() # 调用父类nn.Module的构造函数。
        # 定义一个线性层 (全连接层)。
        # nn.Linear(in_features, out_features) 表示输入维度为in_features，输出维度为out_features。
        # 对于 y = wx + b 形式的线性回归，输入x是一个特征，输出y是一个预测值，所以都是1。
        self.linear = nn.Linear(1, 1) # 输入维度=1，输出维度=1

    def forward(self, x):
        # 定义模型的前向传播逻辑。当模型被调用时（例如 model(x)），这个方法会被执行。
        return self.linear(x) # 将输入x通过线性层进行计算并返回结果。

# 4. 生成训练数据
# 真实关系: y = 2x + 1，我们希望模型学习到这个关系。
x_train = torch.rand(100, 1) * 10 # 生成100个0到10之间的随机数作为输入x。形状 (100, 1)。
# 根据真实关系生成y_train，并添加一些随机噪声，模拟真实世界的数据不完全精确。
y_train = 2 * x_train + 1 + torch.randn(100, 1) * 0.1 # 形状 (100, 1)。
# 将数据移动到指定的设备（CPU或GPU），以便在相应设备上进行计算。
x_train = x_train.to(device)
y_train = y_train.to(device)

# 5. 创建模型、优化器和损失函数
model = SimpleNet().to(device) # 实例化模型，并将其移动到指定设备。
# 定义优化器。优化器负责根据损失函数的梯度来更新模型的参数。
# torch.optim.SGD (Stochastic Gradient Descent) 是一种常见的优化算法。
# model.parameters() 获取模型中所有可训练的参数（w和b）。
# lr (learning rate) 是学习率，控制每次参数更新的步长。
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# 定义损失函数。损失函数衡量模型预测值与真实值之间的差异。
# nn.MSELoss (Mean Squared Error Loss) 均方误差，常用于回归任务。
criterion = nn.MSELoss()

# 6. 训练循环
epochs = 5000 # 定义训练的轮次。

print("\n训练开始...")
for epoch in range(epochs):
    # 前向传播：模型根据当前参数对输入x_train进行预测，得到y_pred。
    y_pred = model(x_train)

    # 计算损失：使用损失函数衡量y_pred和真实值y_train之间的差距。
    loss = criterion(y_pred, y_train)

    # 反向传播：
    optimizer.zero_grad() # 清除之前计算的梯度。在PyTorch中，梯度会累积，所以每次反向传播前需要清零。
    loss.backward() # 执行反向传播，计算损失函数对模型所有可训练参数的梯度。
    optimizer.step() # 根据计算出的梯度和学习率，更新模型的参数（w和b）。这就是“梯度下降”的核心步骤。

    # 每100个epoch打印一次训练状态。
    if (epoch + 1) % 100 == 0:
        # model.linear.weight.item() 和 model.linear.bias.item() 获取线性层的权重w和偏置b的当前值。
        w = model.linear.weight.item()
        b = model.linear.bias.item()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, w: {w:.2f}, b: {b:.2f}')

# 7. 打印最终训练结果
w = model.linear.weight.item()
b = model.linear.bias.item()
print(f'\n训练完成！')
print(f'学习到的函数: y = {w:.2f}x + {b:.2f}')
print(f'实际函数: y = 2.00x + 1.00')

# 8. 测试模型
test_x = torch.tensor([[0.0], [5.0], [10.0]]).to(device) # 创建测试数据，并移动到设备。
# torch.no_grad() 上下文管理器，表示在这个代码块中不计算梯度。
# 在推理或评估阶段，我们不需要计算梯度，这可以节省内存和计算资源。
with torch.no_grad():
    test_y = model(test_x) # 使用训练好的模型进行预测。
    print("\n预测结果：")
    for x_val, y_val in zip(test_x, test_y):
        print(f'x = {x_val.item():.1f}, y = {y_val.item():.2f}')
```

**运行与输出**：
在 PowerShell 中运行：

```powershell
python pytorch_5min.py
```

你将看到类似如下的输出（具体数值可能因设备和随机性略有不同）：

```
Using mps device # 或 Using cuda device / Using cpu device
tensor([3., 6., 9.])
tensor([ 2.,  8., 18.])
tensor(28.)
tensor(28.)
torch.Size([3])

训练开始...
Epoch [100/5000], Loss: 0.0988, w: 2.09, b: 0.41
Epoch [200/5000], Loss: 0.0420, w: 2.05, b: 0.64
...
Epoch [5000/5000], Loss: 0.0066, w: 2.00, b: 1.02

训练完成！
学习到的函数: y = 2.00x + 1.02
实际函数: y = 2.00x + 1.00

预测结果：
x = 0.0, y = 1.02
x = 5.0, y = 11.00
x = 10.0, y = 20.98
```

**核心概念解读**：

1.  **张量 (Tensor) 操作**：
    *   `torch.tensor()`：用于创建 PyTorch 的张量。张量是 PyTorch 中所有数据（包括模型输入、输出、参数）的核心数据结构。它们类似于 NumPy 数组，但提供了 GPU 加速和自动求导功能。
    *   **形状 (Shape)**：张量的 `shape` 属性（如 `x.shape`）描述了张量的维度信息。例如，`torch.Size([3])` 表示一个包含 3 个元素的 1 维张量。
    *   **矩阵乘法 (`torch.matmul` 或 `@`)**：这是深度学习中非常基础且重要的操作。它不仅用于线性代数运算，还常用于张量形状的变换。例如，一个形状为 `(B, T, embd)` 的张量与一个形状为 `(embd, C)` 的张量相乘，结果将是一个形状为 `(B, T, C)` 的张量。这种维度变换在后续的 LLM 模型中会频繁出现。
    *   `.to(device)`：将张量数据移动到指定的计算设备（CPU、GPU 或 MPS）。这对于利用 GPU 加速至关重要。

2.  **模型与神经网络层 (`nn.Module`)**：
    *   `nn.Module`：是 PyTorch 中所有神经网络模块的基类。我们自定义的模型（如 `SimpleNet`）都需要继承它。
    *   `__init__` 方法：用于定义模型的各个层（例如 `nn.Linear`）。这些层内部包含了可训练的参数（权重 `w` 和偏置 `b`）。
    *   `nn.Linear(in_features, out_features)`：一个线性层，执行 `y = xW^T + b` 的操作。它将输入从 `in_features` 维度映射到 `out_features` 维度。在 `SimpleNet` 中，`nn.Linear(1, 1)` 表示输入一个标量，输出一个标量，内部就是 `y = x * w + b`。
        *   你可以通过 `model.linear.weight.item()` 和 `model.linear.bias.item()` 查看 `w` 和 `b` 的值。
    *   `forward` 方法：定义了模型从输入到输出的计算路径（前向传播）。当调用 `model(x)` 时，实际上就是执行 `model.forward(x)`。
    *   `grad_fn`：PyTorch 自动求导机制的关键。当你对张量执行操作时，PyTorch 会构建一个计算图，记录这些操作。`grad_fn` 属性指示了这个张量是如何计算出来的，以便在反向传播时计算梯度。对用户而言，通常不需要直接操作 `grad_fn`。

3.  **反向传播 (Backward Propagation) 与梯度下降 (Gradient Descent)**：
    *   **训练 (Training)** 的核心过程：
        1.  **前向传播 (Forward Pass)**：模型接收输入数据 `x_train`，通过 `forward` 方法计算出预测结果 `y_pred`。
        2.  **计算损失 (Loss Calculation)**：使用**损失函数**（如 `nn.MSELoss`）衡量 `y_pred` 与真实标签 `y_train` 之间的差异，得到一个标量值 `loss`。损失值越小，表示模型预测越准确。
        3.  **反向传播 (Backward Pass)**：
            *   `optimizer.zero_grad()`：在每次反向传播之前，必须将模型参数的梯度清零。因为 PyTorch 默认会累积梯度。
            *   `loss.backward()`：这是触发反向传播的关键。PyTorch 会沿着计算图从 `loss` 反向追溯到模型的每个参数，自动计算 `loss` 对每个参数的**梯度**（即损失函数对参数的偏导数）。梯度指示了如何调整参数才能使损失函数减小最快。
        4.  **参数更新 (Parameter Update)**：
            *   `optimizer.step()`：优化器使用计算出的梯度和预设的**学习率 (learning rate)** 来更新模型的参数。更新规则通常是 `新参数 = 原参数 - 学习率 * 梯度`。
            *   **梯度下降**：这个迭代更新参数以寻找损失函数最小值的过程就叫做梯度下降。学习率决定了每次更新的步长。

    *   PyTorch 的自动求导 (`autograd`) 机制极大地简化了深度学习的实现，我们无需手动计算复杂的导数，只需定义好前向传播，`loss.backward()` 就会自动处理反向传播。

4.  **`torch.no_grad()`**：
    *   在评估或推理阶段，我们不需要计算梯度，因为我们不更新模型参数。使用 `with torch.no_grad():` 可以禁用梯度计算，从而节省内存和计算资源。

---

### 5. 实现一个真正的Bigram模型 (PyTorch版)

现在，我们将把之前“机器学习风格”的 Bigram 模型，用 PyTorch 的张量和模块来实现，使其成为一个真正可训练的深度学习模型。

在直接进入 `babygpt_v1.py` 之前，文档提到了一个中间版本 `simplebigrammodel_torch.py`，它将 `simplebigrammodel.py` 中的原生 Python 列表操作替换为了 PyTorch 张量操作，这是理解从原生 Python 到 PyTorch 转换的关键一步。但为了节省篇幅，我们直接跳到最终的 `babygpt_v1.py`，它包含了完整的 PyTorch 训练和推理流程。

#### 5.1 `babygpt_v1.py` 代码解读

```python
# babygpt_v1.py
import torch # 导入PyTorch库
import torch.nn as nn # 导入神经网络模块，包含各种层
from torch.nn import functional as F # 导入神经网络函数模块，包含激活函数、损失函数等
from typing import List # 用于类型注解
import time # 用于计时

torch.manual_seed(42) # 设置PyTorch的随机数种子，确保结果可复现。

# --- 全局配置参数 ---
prompts = ["春江", "往事"] # 用于模型推理的初始输入字符串
max_new_token = 100 # 模型生成新token的最大数量
max_iters = 5000 # 训练的最大迭代次数（epoch）
eval_iters = 100 # 每次评估时，用于计算平均损失的批次数量
eval_interval = 200 # 每隔多少次迭代进行一次模型评估并打印损失
batch_size = 32 # 每个训练批次中包含的独立序列数量 (B)
block_size = 8 # 每个序列的最大长度，即模型考虑的上下文窗口大小 (T)
learning_rate = 1e-2 # 优化器的学习率
n_embed = 32 # 嵌入层（Embedding Layer）的维度。每个token将被映射到一个n_embed维的向量。
tain_data_ratio = 0.9 # 训练数据占总数据集的比例，剩余部分作为验证数据

# 设备检测：选择可用的计算设备 (GPU或CPU)
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
print(f"Using {device} device") # 打印当前使用的设备

# 读取数据集ci.txt
with open('ci.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenizer 类：与simplebigrammodel.py中的实现完全相同，负责文本的编码和解码。
class Tokenizer:
    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, l: List[int]) -> str:
        return ''.join([self.itos[i] for i in l])

# BabyGPT 类：我们的Bigram语言模型，继承自nn.Module
class BabyGPT(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int):
        # 构造函数，初始化模型层
        super().__init__() # 调用父类nn.Module的构造函数
        # 1. token_embedding_table (嵌入层):
        # nn.Embedding 层是一个查找表，用于将离散的token ID映射到连续的、密集的向量表示。
        # vocab_size: 词汇表的大小，即有多少个唯一的token ID。
        # n_embd: 每个token ID将被映射到的向量的维度（嵌入维度）。
        # 例如，如果token ID是5，它会查找并返回一个长度为n_embd的向量。
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # 2. lm_head (语言模型头):
        # nn.Linear 层是一个全连接层，用于将中间表示映射回词汇表大小的维度。
        # n_embd: 输入维度，来自嵌入层的输出。
        # vocab_size: 输出维度，对应词汇表中每个token的预测得分（logits）。
        # 模型的最终目标是预测下一个token是词汇表中哪个token，所以输出维度必须是vocab_size。
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        # 模型的前向传播方法。
        # idx: 输入的token ID序列批次，形状为 (B, T)。
        # targets: 可选参数，真实的下一个token ID序列批次，形状为 (B, T)。在训练时提供，推理时为None。

        # 1. 嵌入操作: 将输入的token ID转换为嵌入向量。
        # self.token_embedding_table(idx) 会将 idx (B, T) 转换为 tok_emb (B, T, n_embd)。
        # 每一个token ID都被其对应的n_embd维向量替换。
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)

        # 2. 线性投影: 将嵌入向量投影回词汇表维度，得到logits。
        # self.lm_head(tok_emb) 会将 tok_emb (B, T, n_embd) 转换为 logits (B, T, vocab_size)。
        # 每个位置的嵌入向量都被线性层处理，输出一个长度为vocab_size的向量，
        # 这个向量的每个元素代表了下一个token是对应词汇表中token的“得分”。
        logits = self.lm_head(tok_emb) # (B, T, vocab_size)

        loss = None
        # 如果提供了targets，则计算损失。这通常发生在训练阶段。
        if targets is not None:
            # 为了计算交叉熵损失，需要对logits和targets的形状进行调整。
            # F.cross_entropy 函数的第一个参数 (input) 期望形状为 (N, C)，其中N是样本数，C是类别数。
            # 它的第二个参数 (target) 期望形状为 (N)。

            B, T, C = logits.shape # 获取logits的批次大小、序列长度和词汇表大小。
            
            # 将logits的形状从 (B, T, C) 变形为 (B*T, C)。
            # 这个操作并没有丢失信息，只是改变了张量的逻辑视图，将所有B*T个“预测任务”扁平化。
            logits = logits.view(B * T, C)
            
            # 将targets的形状从 (B, T) 变形为 (B*T)。
            # 同样是扁平化，将所有B*T个“真实标签”扁平化。
            targets = targets.view(B * T)
            
            # 计算交叉熵损失。
            # 交叉熵损失常用于分类问题，衡量模型预测的概率分布与真实标签之间的差异。
            # 对于语言模型，它衡量模型预测的下一个token的概率分布与真实下一个token的“独热编码”之间的差异。
            loss = F.cross_entropy(logits, targets)

        return logits, loss # 返回logits和损失。

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # 序列生成方法。
        # idx: 初始的prompt token ID序列批次，形状为 (B, T)。
        # max_new_tokens: 需要生成的新token的最大数量。

        # 循环生成 max_new_tokens 个新token。
        for _ in range(max_new_tokens):
            # 在Bigram模型中，我们只关心序列的最后一个token来预测下一个。
            # 但在这里，为了与后续更复杂的模型（如Transformer）保持接口一致，
            # 我们仍然传入整个idx序列。然而，在实际处理时，我们只取最后一个token的嵌入。
            # 这里idx的形状是 (B, T_current)，其中 T_current 会随着生成不断增长。
            
            # 调用模型的forward方法，获取当前序列批次的logits。
            # 注意：这里只传idx，因为是推理模式，不需要targets，所以loss会是None。
            # logits的形状是 (B, T_current, vocab_size)。
            logits, _ = self(idx)

            # 提取每个序列中最后一个token的logits。
            # logits[:, -1, :] 表示取所有批次 (:) 的最后一个序列位置 (-1) 的所有词汇表维度 (:)。
            # 结果形状是 (B, vocab_size)。
            logits = logits[:, -1, :]

            # 使用Softmax函数将logits转换为概率分布。
            # F.softmax(input, dim) 会在指定维度上进行Softmax操作。
            # dim=-1 表示对最后一个维度（即vocab_size维度）进行Softmax，
            # 确保每个token的预测概率和为1。
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)

            # 根据概率分布随机采样下一个token ID。
            # torch.multinomial(input, num_samples) 从input的行中（将其视为概率分布）抽取num_samples个索引。
            # 这里input是probs (B, vocab_size)，我们从每个批次的概率分布中采样一个token。
            # 结果形状是 (B, 1)。
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # 将新生成的token ID拼接到当前序列的末尾。
            # torch.cat((tensor1, tensor2), dim) 沿着指定维度拼接张量。
            # dim=1 表示沿着序列长度维度拼接，将idx_next (B, 1) 拼接到idx (B, T_current) 后面。
            # 结果形状是 (B, T_current + 1)。
            idx = torch.cat((idx, idx_next), dim=1)
        return idx # 返回包含生成token的完整序列批次。

# --- 数据准备 ---
tokenizer = Tokenizer(text) # 实例化分词器
vocab_size = tokenizer.vocab_size # 获取词汇表大小
# 将整个文本编码为整数ID序列，并转换为PyTorch的long类型张量，然后移动到指定设备。
# torch.long 是整数类型，通常用于表示索引或分类标签。
raw_data = torch.tensor(tokenizer.encode(text), dtype=torch.long).to(device)

# 划分训练集和验证集
n = int(tain_data_ratio * len(raw_data)) # 计算训练集的大小
data = {'train': raw_data[:n], 'val': raw_data[n:]} # 划分数据

# get_batch 函数：用于获取批次数据，PyTorch版本
def get_batch(data_split: torch.Tensor, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    从给定的数据分割（训练集或验证集）中随机获取一批数据x和y。
    x和y都是PyTorch张量，形状分别为 (B, T) 和 (B, T)。
    y中的每个值都是x中对应位置值的下一个值。
    '''
    # 随机选择 batch_size 个起始索引。
    # torch.randint(high, size) 生成 size 形状的张量，其元素在 [0, high) 范围内。
    # len(data_split) - block_size 确保有足够的空间截取 block_size 长度的序列。
    ix = torch.randint(len(data_split) - block_size, (batch_size,)) # (batch_size,)

    # 使用torch.stack将多个序列堆叠成一个批次张量。
    # data_split[i:i+block_size] 截取单个序列。
    # torch.stack([..., ..., ...]) 将这些序列沿新维度堆叠。
    x = torch.stack([data_split[i:i+block_size] for i in ix]) # (B, T)
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix]) # (B, T)
    
    # 将批次数据移动到指定设备（如果尚未移动）。
    x, y = x.to(device), y.to(device)
    return x, y

# estimate_loss 函数：用于在训练和验证集上评估模型损失
@torch.no_grad() # 装饰器：表示在此函数中不需要计算梯度，节省内存和计算。
def estimate_loss(model: nn.Module, data: dict[str, torch.Tensor], batch_size: int, block_size: int, eval_iters: int) -> dict[str, float]:
    '''
    计算模型在训练集和验证集上的平均损失。
    model: 待评估的模型。
    data: 包含'train'和'val'数据的字典。
    eval_iters: 用于计算平均损失的批次数量。
    '''
    out = {}
    model.eval() # 将模型设置为评估模式。
                  # 在评估模式下，某些层（如Dropout、BatchNorm）的行为会发生改变，例如Dropout会关闭。
    for split in ['train', 'val']: # 遍历训练集和验证集
        losses = torch.zeros(eval_iters, device=device) # 初始化一个张量来存储每次评估的损失
        for k in range(eval_iters): # 循环eval_iters次，获取多个批次来计算平均损失
            x, y = get_batch(data[split], batch_size, block_size) # 获取一批数据
            _, loss = model(x, y) # 前向传播，计算损失
            losses[k] = loss.item() # 存储损失值 (loss.item() 将张量转换为Python标量)
        out[split] = losses.mean().item() # 计算平均损失并存储
    model.train() # 评估完成后，将模型切换回训练模式。
    return out

# --- 模型实例化与训练 ---
model = BabyGPT(vocab_size, n_embed).to(device) # 实例化模型，并将其移动到指定设备。
# 计算模型参数量，用于了解模型大小。
# model.parameters() 返回一个迭代器，包含模型所有可训练的参数张量。
num_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {num_params}")

# 定义优化器。
# torch.optim.AdamW 是一种常用的优化算法，比SGD在许多情况下表现更好。
# model.parameters() 获取模型所有可训练的参数。
# lr=learning_rate 设置学习率。
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\n开始训练...")
start_time = time.time() # 记录训练开始时间
tokens_processed = 0 # 统计已处理的token数量

for iter in range(max_iters): # 训练循环
    # 每隔eval_interval次迭代，进行一次评估并打印损失。
    if iter % eval_interval == 0:
        elapsed = time.time() - start_time
        tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0 # 计算每秒处理的token数
        losses = estimate_loss(model, data, batch_size, block_size, eval_iters) # 评估损失
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, speed: {tokens_per_sec:.2f} tokens/sec")

    # 获取一个训练批次的数据
    x, y = get_batch(data['train'], batch_size, block_size)

    # 前向传播：计算模型的logits和损失。
    logits, loss = model(x, y)

    # 反向传播与参数更新：
    optimizer.zero_grad(set_to_none=True) # 清除旧梯度，set_to_none=True更高效地释放内存
    loss.backward() # 计算损失对模型参数的梯度
    optimizer.step() # 根据梯度更新模型参数

    tokens_processed += batch_size * block_size # 更新已处理token计数

elapsed_time = time.time() - start_time
print(f"\n训练完成！总耗时: {elapsed_time:.2f} 秒")

# --- 模型推理 (文本生成) ---
print("\n开始推理（生成文本）...")
# 将prompt字符串编码为token ID，转换为PyTorch张量，并堆叠成批次。
# prompts 是一个字符串列表，需要先encode成List[int]，再转换为torch.tensor。
# torch.stack([..., ..., ...]) 将多个一维张量堆叠成一个二维张量 (B, T)。
prompt_tokens = torch.stack([torch.tensor(tokenizer.encode(p), dtype=torch.long).to(device) for p in prompts])

# 调用模型的generate方法生成新token。
# model.eval() 确保在推理时模型处于评估模式（例如禁用Dropout）。
model.eval()
result = model.generate(prompt_tokens, max_new_token)

# 解码并打印生成的结果
print("\n生成结果：")
for tokens_list_tensor in result:
    # tokens_list_tensor 是一个PyTorch张量，需要先转换为Python列表 (tolist()) 再解码。
    print(tokenizer.decode(tokens_list_tensor.tolist()))
    print('-'*10)
```

**运行与输出**：
在 PowerShell 中运行：

```powershell
python babygpt_v1.py
```

你将看到类似如下的输出（具体数值和生成文本会因随机性和训练轮次有所不同）：

```
Using mps device # 或 Using cuda device / Using cpu device
模型参数量: 399620

开始训练...
step 0: train loss 8.9236, val loss 8.9194, speed: 1118.03 tokens/sec
step 200: train loss 5.8334, val loss 5.9927, speed: 50238.47 tokens/sec
step 400: train loss 5.5678, val loss 5.7631, speed: 56604.35 tokens/sec
...
step 4800: train loss 5.1789, val loss 5.4211, speed: 62000.00 tokens/sec (示例)

训练完成！总耗时: 约几十秒钟

开始推理（生成文本）...

生成结果：
春江花月夜。
我不知。
月在天涯。
千里江山如画。
一片冰心在玉壶。
不堪回首。
风流总被雨打风吹去。
人生长恨水长东。
一枝红杏出墙来。
海棠依旧。
绿肥----------
往事不堪回首。
人去楼空。
月明中。
风雨故人来。
一夜风雨。
花落知多少。
不堪回首。
风流总被雨打风吹去。
人生长恨水长东。
一枝红杏出墙来。
海棠依旧。
绿肥----------
```

**核心概念解读 (`babygpt_v1.py`)**：

1.  **模型结构 (`BabyGPT` 类)**：
    *   `BabyGPT` 继承自 `nn.Module`，这是 PyTorch 模型的基本要求。
    *   **`nn.Embedding(vocab_size, n_embd)` (嵌入层)**：
        *   **原理**：嵌入层是一个查找表，将离散的整数 ID（token）映射到一个连续的、低维度的实数向量（嵌入向量）。例如，token ID `5` 不仅仅是一个数字，它现在被表示为一个 `n_embd` 维的向量，这个向量捕获了该 token 的语义信息。
        *   **作用**：在自然语言处理中，直接使用 token ID 进行计算效率低下，且无法捕捉词语之间的相似性。嵌入层解决了这个问题，它将每个 token 转换为一个稠密的向量表示，使得语义相似的词在向量空间中距离更近。这是所有现代 LLM 的第一层。
        *   **示例**：`self.token_embedding_table(idx)`。如果 `idx` 是 `[1, 5, 2]`，`n_embd` 是 `32`，那么输出将是 `[[emb_vec_for_1], [emb_vec_for_5], [emb_vec_for_2]]`，其中每个 `emb_vec` 都是一个 32 维的向量。
        *   **参数量**：嵌入层内部有一个 `(vocab_size, n_embd)` 大小的权重矩阵，每个元素都是一个可学习的参数。
    *   **`nn.Linear(n_embd, vocab_size)` (`lm_head` 语言模型头)**：
        *   **原理**：这是一个全连接层，将模型的中间表示（这里是嵌入层的输出 `tok_emb`）再次投影回 `vocab_size` 维度。
        *   **作用**：模型的最终目标是预测下一个 token 是词汇表中的哪个词。`lm_head` 层将 `n_embd` 维的嵌入向量转换为一个 `vocab_size` 维的向量，这个向量的每个元素代表了词汇表中对应 token 的“得分”或“logit”。得分越高，模型认为该 token 作为下一个词的可能性越大。
        *   **参数量**：线性层内部有一个 `(vocab_size, n_embd)` 大小的权重矩阵和一个 `(vocab_size,)` 大小的偏置向量。
    *   **`forward` 方法中的计算流**：
        1.  `idx (B, T)` -> `token_embedding_table` -> `tok_emb (B, T, n_embd)`：将输入的 token ID 序列转换为它们的嵌入向量序列。
        2.  `tok_emb (B, T, n_embd)` -> `lm_head` -> `logits (B, T, vocab_size)`：将嵌入向量序列投影回 `vocab_size` 维度，得到每个位置的下一个 token 的预测得分。
    *   **`generate` 方法中的 Softmax 和 `torch.multinomial`**：
        1.  `logits = logits[:, -1, :]`：在 Bigram 模型中，我们只关心序列中最后一个 token 的预测结果，所以只取 `logits` 的最后一个时间步。
        2.  `probs = F.softmax(logits, dim=-1)`：
            *   **Softmax 函数**：将原始的 `logits`（可以是任意实数）转换为一个概率分布，使得所有元素的和为 1，且每个元素都在 0 到 1 之间。它通常用于多分类任务的最后一层。
            *   **`dim=-1`**：指定在哪个维度上进行 Softmax 操作。这里 `dim=-1` 表示对 `vocab_size` 维度进行操作，确保每个 token 的预测概率之和为 1。
        3.  `idx_next = torch.multinomial(probs, num_samples=1)`：
            *   **`torch.multinomial`**：根据给定的概率分布进行随机采样。它会从 `probs` 的每一行（视为一个概率分布）中抽取 `num_samples` 个索引。
            *   **作用**：这是实现文本生成中“随机性”的关键。模型不是直接选择概率最高的词，而是根据概率分布进行随机抽取，这使得生成的文本更加多样和自然，避免了重复和僵化。

2.  **损失函数 (`F.cross_entropy`)**：
    *   **原理**：交叉熵损失函数是分类任务中最常用的损失函数之一。它衡量了模型预测的概率分布与真实标签的“独热编码”之间的差异。差异越大，损失值越高。
    *   **形状要求**：`F.cross_entropy` 要求输入 `logits` 的形状为 `(N, C)` (N: 样本数, C: 类别数)，`targets` 的形状为 `(N)` (N: 样本数，包含类别索引)。
    *   **数据重塑**：
        *   `logits = logits.view(B*T, C)`：将 `(B, T, C)` 形状的 `logits` 扁平化为 `(B*T, C)`。这相当于把所有 `B*T` 个预测任务（每个任务预测一个下一个 token）看作独立的样本。
        *   `targets = targets.view(B*T)`：将 `(B, T)` 形状的 `targets` 扁平化为 `(B*T)`，与 `logits` 的样本数对应。
    *   **作用**：在训练过程中，交叉熵损失指导模型调整参数，使其预测的下一个 token 的概率分布尽可能接近真实的下一个 token。

3.  **优化器 (`torch.optim.AdamW`)**：
    *   **原理**：`AdamW` 是 `Adam` 优化器的一个变种，它在权重衰减（L2 正则化）的处理上有所不同，通常在许多深度学习任务中表现优于传统的 `SGD`。
    *   **作用**：优化器负责根据损失函数计算出的梯度来更新模型的参数（`nn.Embedding` 和 `nn.Linear` 中的权重）。它通过迭代地调整参数，使损失函数最小化。

4.  **数据处理 (`get_batch` PyTorch 版)**：
    *   `raw_data = torch.tensor(tokenizer.encode(text), dtype=torch.long).to(device)`：将原始文本编码后的 Python 列表转换为 PyTorch 的 `LongTensor`，并移动到设备上。`dtype=torch.long` 非常重要，因为 token ID 是整数，并且作为索引使用。
    *   `ix = torch.randint(len(data_split) - block_size, (batch_size,))`：使用 `torch.randint` 替代 `random.choices` 来生成随机索引，直接生成 PyTorch 张量。
    *   `x = torch.stack([...])` 和 `y = torch.stack([...])`：使用 `torch.stack` 将多个独立的序列张量堆叠成一个批次张量，形状从 `(block_size,)` 变为 `(batch_size, block_size)`。这是创建批处理数据张量的常用方法。

5.  **模型评估 (`estimate_loss` 函数)**：
    *   `@torch.no_grad()`：确保在评估时不会计算梯度，节省资源。
    *   `model.eval()`：将模型设置为评估模式。这会影响一些层的行为，例如 `nn.Dropout` 层在训练时会随机丢弃神经元以防止过拟合，但在评估时会关闭，以确保预测的一致性。
    *   `model.train()`：评估完成后，将模型切换回训练模式。
    *   **作用**：定期评估模型在训练集和验证集上的损失，可以帮助我们监控模型是否过拟合（训练损失低但验证损失高）或欠拟合（训练和验证损失都高），从而调整模型或训练策略。

6.  **参数量计算**：
    *   `sum(p.numel() for p in model.parameters())`：计算模型中所有可训练参数的总数量。
    *   **`BabyGPT` 的参数量**：
        *   `Embedding` 层：`vocab_size * n_embd`
        *   `Linear` 层（`lm_head`）：`n_embd * vocab_size` (权重) + `vocab_size` (偏置)
        *   总参数量 = `vocab_size * n_embd + n_embd * vocab_size + vocab_size`
        *   在我们的例子中：`6148 * 32 + 32 * 6148 + 6148 = 196736 + 196736 + 6148 = 399620`。
        *   每个参数通常是 4 字节的浮点数（`float32`），所以模型大小约为 `399620 * 4 字节 = 1.59 MB`。这个模型非常小，被称为“0.0004B”参数的模型（0.5B = 5亿参数）。

### 6. 回顾与下一步

**回顾**：
到目前为止，我们已经用大约 130 行 Python 代码实现了一个基于 PyTorch 的 Bigram 语言模型。这个模型能够生成看起来像是诗词的文本，并且我们亲身体验了：

*   **Tokenizer**：将文本转换为数字序列。
*   **Embedding 层**：将离散的 token 映射为连续的向量表示。
*   **Linear 层 (lm_head)**：将模型的中间表示投影回词汇表维度，得到预测得分。
*   **模型参数 (Parameters)**：Embedding 和 Linear 层内部的可学习权重和偏置。
*   **前向传播 (Forward Pass)**：模型从输入计算到输出的过程。
*   **损失函数 (Loss Function)**：衡量模型预测与真实值之间差异的工具 (`F.cross_entropy`)。
*   **反向传播 (Backward Pass)**：计算损失对模型参数的梯度。
*   **优化器 (Optimizer)**：根据梯度更新模型参数的算法 (`AdamW`)。
*   **学习率 (Learning Rate)**：控制参数更新步长的超参数。
*   **推理 (Inference) / 生成 (Generation)**：使用训练好的模型生成新文本。
*   **批处理 (Batching) 与张量 (Tensors)**：为高效计算准备数据结构。

这个 Bigram 模型虽然简单，但它麻雀虽小五脏俱全，涵盖了深度学习语言模型的基本流程。它最大的局限是只考虑了前一个词的上下文。

**下一步**：
在下一篇文章中，我们将基于 `babygpt_v1.py` 的基础，引入 **“自注意力机制 (Self-Attention Mechanism)”**，这是 Transformer 模型的核心，也是现代 LLM 能够理解长距离依赖的关键。通过实现自注意力，我们将逐步构建一个完整的 GPT 模型。

**建议**：
如果你对模型流程和结构仍有疑问，可以尝试修改 `babygpt_v1.py` 中的参数（例如 `n_embed`、`block_size`、`learning_rate`），观察它们对训练和生成结果的影响。
如果你对中间各种变量的转换不理解，强烈建议在调试器中（如 VS Code 的 Jupyter Notebook）逐行运行代码，并使用 `.shape` 属性观察每个张量的形状变化，使用 `.item()` 或打印张量本身来查看其内容，这将帮助你深入理解其中的细节。

Happy Hacking！



