# nlp_text_summarization
这是从transformers开始的文本摘要尝试，在该项目中，我会先从预训练微调开始，针对中文文本，尝试多种可能的预训练模型。
然后逐渐探索其他算法或模型，之后会逐一放在该项目中。
目前尝试的预训练模型：1.mt5,2.bart-base-chinese。
环境配置：python:3.10.9  pytorch:2.0.0+cu118  transformers:4.36.2
使用数据集：大规模中文短文本摘要语料库 LCSTS，该语料基于新浪微博短新闻构建，规模超过 200 万。
以上代码引用自：https://transformers.run/c3/2022-03-29-transformers-note-8/#1-%E5%87%86%E5%A4%87%E6%95%B0%E6%8D%AE
