# 需要先安装: pip install ijson
import json

import ijson
from collections import defaultdict

filename = "test_data/turnonlights/test_turnonlights_251029.json"
all_records = []

try:
    with open(filename, 'r') as file:
        # 使用 ijson 的 items() 方法来查找和解析所有的顶级数组元素
        # 'item' 会遍历文件中的每个顶层 JSON 元素
        # 在这种结构下，每个顶层元素都是一个完整的数组
        for array in ijson.items(file, '', multiple_values=True):
            if isinstance(array, list):
                all_records.extend(array)
    print(f"成功解析并合并了 {len(all_records)} 条记录。")

except FileNotFoundError:
    print(f"错误：文件 {filename} 未找到。")
except Exception as e:
    print(f"解析过程中发生错误: {e}")

all_records = sorted(all_records, key=lambda x: x['level'])
with open(f"test_data/turnonlights/test_turnonlights.json",'w') as file:
    json.dump(all_records, file, indent=4)
