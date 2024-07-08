import os
import glob

d = glob.glob("./videos/output/*")
for i in d:
    t = os.path.splitext(os.path.split(i)[1])
    print(t[0]+"_infered"+t[1])