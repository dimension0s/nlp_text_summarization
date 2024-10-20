import sentencepiece as spm
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,T5Tokenizer

# 1.尝试加载慢性分词器：T5Tokenizer
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,T5Tokenizer
model_path = "E:\\offline_model\\csebuetnlpmT5_multilingual_XLSum"
try:
    # 尝试加载慢速分词器
    tokenizer = T5Tokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    print("慢速分词器和模型加载成功！")
except Exception as e:
    print("加载慢速分词器或模型时出错：", e)

# 2.尝试加载快速分词器：AutoTokenizer
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,T5Tokenizer
model_path = "E:\\offline_model\\csebuetnlpmT5_multilingual_XLSum"
try:
    # 尝试加载慢速分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    print("快速分词器和模型加载成功！")
except Exception as e:
    print("加载快速分词器或模型时出错：", e)

