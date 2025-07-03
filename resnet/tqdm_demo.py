import time
from tqdm import tqdm

# 总的迭代次数
total_steps = 50

# 用tqdm创建进度条
for _ in tqdm(range(total_steps),desc='Loading'):
    time.sleep(0.3)