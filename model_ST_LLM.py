# ------------------------------------------------------------------
# 导入所需的PyTorch库
# ------------------------------------------------------------------
import torch
import torch.nn as nn
# 从Hugging Face Transformers库中导入预训练的GPT-2模型结构
from transformers.models.gpt2.modeling_gpt2 import GPT2Model


# ------------------------------------------------------------------
# 时间嵌入模块 (TemporalEmbedding)
# 作用：将离散的时间信息（如一天中的哪个时刻，一周中的星期几）转换成模型可以理解的连续向量。
# ------------------------------------------------------------------
class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        """
        初始化时间嵌入层。
        :param time: int, 一天中的时间步总数 (例如 24小时 * 12个5分钟 = 288)
        :param features: int, 每个时间嵌入向量的维度
        """
        super(TemporalEmbedding, self).__init__()
        self.time = time
        # 创建“日内时间”嵌入矩阵，每一行代表一天中一个时刻的向量
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)  # 使用xavier方法初始化，有助于模型训练

        # 创建“周内时间”嵌入矩阵，每一行代表星期一到星期日的向量
        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        """
        前向传播，根据输入的具体时间信息，查找对应的嵌入向量。
        :param x: 输入数据，其中包含了日和周的时间戳
        :return: 组合后的时间嵌入向量
        """
        # --- 获取日内时间嵌入 ---
        # 从输入数据中提取出“日内”时间戳
        day_emb = x[..., 1]
        # 将时间戳转换为整数索引，并从日内嵌入矩阵中查找对应的向量
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]
        # 调整向量维度以方便后续计算
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        # --- 获取周内时间嵌入 ---
        # 从输入数据中提取出“周内”时间戳 (星期几)
        week_emb = x[..., 2]
        # 查找对应的星期向量
        time_week = self.time_week[
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ]
        # 调整向量维度
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        # --- 组合时间嵌入 ---
        # 将“日内”和“周内”的嵌入向量相加，融合成最终的时间信息
        tem_emb = time_day + time_week
        return tem_emb


# ------------------------------------------------------------------
# GPT-2适配器模块 (PFA - Prompt-based Finetuning Adapter)
# 作用：封装并改造一个预训练的GPT-2模型。它并非完全训练整个GPT-2，而是“冻结”大部分
#      参数，只微调（finetune）其中一小部分，使其在保留强大通用能力的同时，能适应我们
#      特定的时空预测任务，这样做可以大大节省计算资源并防止过拟合。
# ------------------------------------------------------------------
class PFA(nn.Module):
    def __init__(self, device="cuda:0", gpt_layers=6, U=1):
        """
        :param gpt_layers: int, 我们要使用GPT-2的前几层。
        :param U: int, 我们要微调（解冻）最顶部的几层中的非MLP部分。
        """
        super(PFA, self).__init__()
        # 加载预训练的GPT-2模型
        self.gpt2 = GPT2Model.from_pretrained(
            "gpt2", output_attentions=True, output_hidden_states=True
        )
        # 只保留我们需要的层数
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.U = U

        # --- 参数冻结与解冻策略 ---
        # 遍历GPT-2的每一层，决定哪些参数需要训练，哪些不需要
        for layer_index, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():
                # 对底部的层 (非微调层)
                if layer_index < gpt_layers - self.U:
                    # 只允许训练层归一化(ln)和位置编码(wpe)参数，它们对适应新数据分布很重要
                    if "ln" in name or "wpe" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False # 其他参数全部冻结
                # 对顶部的U层 (微调层)
                else:
                    # 冻结其中的MLP（多层感知机）部分，因为这部分通常更具任务特异性
                    if "mlp" in name:
                        param.requires_grad = False
                    # 解冻其他部分（主要是注意力机制），让模型学习新的关联模式
                    else:
                        param.requires_grad = True

    def forward(self, x):
        """
        前向传播，将输入数据通过改造后的GPT-2模型。
        :param x: 融合了时空信息的输入嵌入
        :return: GPT-2处理后的深层特征
        """
        # `inputs_embeds`允许我们直接输入向量而不是文字ID
        return self.gpt2(inputs_embeds=x).last_hidden_state


# ------------------------------------------------------------------
# 时空大型语言模型 (ST-LLM)
# 作用：整个模型的核心，它将所有模块（时间嵌入、空间嵌入、GPT-2）组合在一起，
#      完成从输入历史数据到输出未来预测的完整流程。
# ------------------------------------------------------------------
class ST_LLM(nn.Module):
    def __init__(
        self,
        input_dim=3,     # 每个数据点的原始特征维度 (例如：速度、时间、星期)
        channels=64,     # 模型内部处理时使用的特征维度（通道数）
        num_nodes=170,   # 交通网络中的节点数 (例如：传感器数量)
        input_len=12,    # 输入的历史时间步长度 (例如：用过去1小时的数据)
        output_len=12,   # 需要预测的未来时间步长度 (例如：预测未来1小时的数据)
        llm_layer=6,     # 使用的GPT-2层数
        U=1,             # PFA中微调的层数
        device="cuda:7"
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.output_len = output_len

        # --- 根据数据集确定时间分辨率 ---
        if num_nodes == 170 or num_nodes == 307:
            time = 288  # PEMS-BAY, METR-LA 数据集，5分钟一个点
        elif num_nodes == 250 or num_nodes == 266:
            time = 48   # PEMSD7(M), PEMSD7(L) 数据集，30分钟一个点

        # --- 定义模型各层 ---
        gpt_channel = 256      # 统一嵌入后的特征维度
        to_gpt_channel = 768   # 送入GPT-2模型前的特征维度 (必须是768，因为这是GPT-2的默认维度)

        # 1. 时间嵌入层
        self.Temb = TemporalEmbedding(time, gpt_channel)

        # 2. 空间嵌入层 (节点嵌入)
        # 为每个交通节点（传感器）创建一个可学习的向量，代表其地理空间特征
        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, gpt_channel))
        nn.init.xavier_uniform_(self.node_emb)

        # 3. 初始卷积层
        # 将原始的、多时间步的输入数据压缩并映射到统一的特征维度(gpt_channel)
        self.start_conv = nn.Conv2d(self.input_dim * self.input_len, gpt_channel, kernel_size=(1, 1))

        # 4. 特征融合层
        # 将数据特征、时间特征、空间特征这三者合并，并映射到GPT-2需要的输入维度(to_gpt_channel)
        self.feature_fusion = nn.Conv2d(gpt_channel * 3, to_gpt_channel, kernel_size=(1, 1))

        # 5. GPT-2核心处理层
        self.gpt = PFA(device=device, gpt_layers=llm_layer, U=U)

        # 6. 回归输出层
        # 将经过GPT-2处理后的深层特征，转换为最终的预测结果
        self.regression_layer = nn.Conv2d(to_gpt_channel, self.output_len, kernel_size=(1, 1))

    def forward(self, history_data):
        """
        模型的核心计算流程。
        :param history_data: [批次大小, 历史长度, 节点数, 特征数]
        :return: 预测结果 [批次大小, 1, 节点数, 预测长度]
        """
        input_data = history_data
        batch_size, _, num_nodes, _ = input_data.shape

        # --- 1. 生成时间和空间嵌入 ---
        # 调整数据维度以方便提取时间信息
        tem_emb = self.Temb(history_data.permute(0, 3, 2, 1))
        # 扩展节点嵌入以匹配批次中每个样本
        node_emb = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)

        # --- 2. 处理原始输入数据 ---
        # 调整维度，并将历史时间步和特征合并，准备进行卷积
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        # 通过初始卷积层，提取初步特征
        input_data = self.start_conv(input_data)

        # --- 3. 融合所有信息 ---
        # 将数据特征、时间嵌入、空间嵌入沿特征维度拼接在一起
        data_st = torch.cat([input_data, tem_emb, node_emb], dim=1)
        # 通过融合层，将拼接后的信息整合成一个统一的、高维的表示
        data_st = self.feature_fusion(data_st)

        # --- 4. 通过GPT-2进行深层处理 ---
        # 调整数据维度以符合GPT-2的输入要求 [批次, 序列(节点), 特征]
        data_st = data_st.permute(0, 2, 1, 3).squeeze(-1)
        # 送入GPT-2模型，学习时空数据中的复杂依赖关系
        data_st = self.gpt(data_st)
        # 再次调整维度，准备输出
        data_st = data_st.permute(0, 2, 1).unsqueeze(-1)

        # --- 5. 生成最终预测 ---
        # 通过回归层，将GPT-2输出的深层特征映射为未来12个时间步的预测值
        prediction = self.regression_layer(data_st)

        return prediction
