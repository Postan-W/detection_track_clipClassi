import random
def f1():
    # 把已标注的climb都删了，重新标
    input_path = "./train_data/train.txt"
    output_path = "train_data/train.txt"
    with open(input_path, "r") as f_in:
        with open(output_path, "w") as f_out:
            for line in f_in:
                t = line.strip().split(",")
                if t[0] != "climb":
                    f_out.write(line)

def f2():
    input_path = "./train_data/train.txt"
    output_path = "train_data/train.txt"
    with open(input_path, "r") as f_in:
        with open(output_path, "w") as f_out:
            a = f_in.readlines()
            random.shuffle(a)
            for line in a:
                f_out.write(line)

def f3():
    # 把climb和fall的单独拿出来
    input_path = "./train_data/merged.txt"
    output_path = "train_data/only_climb_fall.txt"
    with open(input_path, "r") as f_in:
        with open(output_path, "w") as f_out:
            for line in f_in:
                t = line.strip().split(",")
                if (t[0] == "climb") or (t[0] == "fall"):
                    f_out.write(line)


