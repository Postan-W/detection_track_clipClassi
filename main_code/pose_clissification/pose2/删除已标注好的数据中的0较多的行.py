input_path = "train_data/merged.txt"
output_path = "train_data/merged.txt"
#超过4个关键点被遮挡的我就不要该行(也就是8个0,一般都是成对出现的)
with open(input_path, "r") as f_in:
    with open(output_path, "w") as f_out:
        count = 0
        for line in f_in:
            c = 0
            t = [float(i) for i in line.strip().split(",")[1:]]
            for num in t:
                if num == 0:
                    c += 1

            if sum(t[:4]) == 0:
                print("头部被遮挡")
                count += 1
                continue

            if c >= 8:
                print("遮挡超过4个关键点")
                count += 1
                continue

            else:
                f_out.write(line)

        print("共筛掉{}无效数据".format(count))


