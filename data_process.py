# 1.构建数据集
# 1.1） 加载数据集
# 由于原数据集太多，在这里截取部分数据用于训练该任务
from torch.utils.data import Dataset

max_dataset_size = 300000

class LCSTS(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file,'rt',encoding='utf-8') as f:
            for idx,line in enumerate(f):
                if idx>=max_dataset_size:
                    break
                items = line.strip().split('!=!')
                assert len(items) == 2
                Data[idx] = {
                    'title': items[0],
                    'content': items[1],
                }
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = LCSTS("data/data1.txt")
valid_data = LCSTS("data/data2.txt")
test_data = LCSTS("data/data3.txt")

train_set_size = len(train_data)  # 300000
valid_set_size = len(valid_data)  # 10666
test_set_size = len(test_data)  # 1106

# 打印测试
print(next(iter(train_data)))
# {'title': '修改后的立法法全文公布', 'content': '新华社受权于18日全文播发修改后的《中华人民共和国立法法》，修改后的立法法
# 分为“总则”“法律”“行政法规”“地方性法规、自治条例和单行条例、规章”“适用与备案审查”“附则”等6章，共计105条。'}
