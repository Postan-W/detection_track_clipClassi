from tqdm import tqdm
import time

l = range(10)
for i in tqdm(l):
    time.sleep(1)
    print(i)