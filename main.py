import My_DivMark_CN
import csv
import time

# 计算起始时间
start = time.time()
# 读取数据，生成语言模型
test = My_DivMark_CN.DivMark_CN("divide_data.txt", "mark_data.txt")
test.init_run()

# 读取测试数据
# test_sentences中元素为句子
# div_ground_truth中的元素为正确分词的列表，mark_ground_truth的元素为正确词性标注的列表
data_f = open("mark_data.csv", "r", encoding='UTF-8')
data_reader = csv.reader(data_f)
test_sentences = []
div_ground_truth = []
mark_ground_truth = []

# 遍历测试数据的每一行
for line in data_reader:
    sentence = ""
    div_gt_list = []
    mark_gt_list = []
    for item in line:
        # 跳过每一行开头的时间序号
        if item == line[0]:
            continue
        # 对元素进行切分，得到词语和词性
        word = item.split('/')[0].split('(')[0].split('{')[0].split('[')[0]
        word_type = item.split('/')[-1].split(')')[0].split('}')[0].split('!')[0].split(']')[0]
        # sentence为句子字符串
        # div_gt_list列表中元素存储词语，mark_gt_list列表中元素存储词性
        sentence += word.strip()
        div_gt_list.append(word)
        mark_gt_list.append(word_type)
        # 句子结束，跳出循环
        if word in ['。', '!', '?']:
            break
    # 生成句子列表和分词、词性标注的正确结果
    test_sentences.append(sentence)
    div_ground_truth.append(div_gt_list)
    mark_ground_truth.append(mark_gt_list)

# 分词并进行词性标注，对结果进行评价
test.evaluate(test_sentences, div_ground_truth, mark_ground_truth)

# 计算总运行时间
end = time.time()
print("-> time :  {:.2f} s".format(end - start))
print("-> total :  {} sentences".format(len(test_sentences)))