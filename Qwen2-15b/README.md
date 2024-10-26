简述：调用Qwen2-1.5b-instruct为自回归模型（来自阿里研发），使用数据集仍为LCSTS中短文本摘要数据集，基于预训练模型做微调训练。
1.模型介绍：
Qwen2-1.5b-instruct，来自阿里研发，详细信息可参考hugging face：https://huggingface.co/Qwen/Qwen2-1.5B-Instruct
和官方文档：https://modelscope.cn/models/qwen/Qwen2-1.5B-Instruct
2.使用的数据集：仍为LCSTS中短文本摘要数据集
3.期间遇到的问题：
1)在数据集的处理中，需要注意的是:令输入和输出都保持一样的长度，从而避免输入输出维度不匹配的问题，该模型使用的是交叉熵损失计算
2)对于内存不足的问题，首先可以尝试将batch_size减小
