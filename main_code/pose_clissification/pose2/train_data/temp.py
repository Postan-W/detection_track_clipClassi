import random


def f():
    # 打乱一下数据
    input_path = "./four_classes_su_plus_jinan.txt"
    output_path = ""
    with open(input_path, "r") as f_in:
        with open(output_path, "w") as f_out:
            lines = f_in.readlines()
            random.shuffle(lines)
            for line in lines:
                f_out.write(line)

