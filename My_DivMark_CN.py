import csv


class DivMark_CN:

    def __init__(self, div_txtpath, mark_txtpath):
        # 初始化语料数据文件路径
        self.div_txtpath = div_txtpath
        self.mark_txtpath = mark_txtpath

        # 创建同名.csv文件
        temp = div_txtpath.split('.')
        self.div_csvpath = str(temp[0]) + ".csv"

        temp = mark_txtpath.split('.')
        self.mark_csvpath = str(temp[0]) + ".csv"

        self.div_sentences = []
        self.mark_sentences = []
        self.marktype_cnt = dict()
        self.unigram_dic = dict()
        self.bigram_dic = dict()
        self.type_transfer = dict()
        self.type_P = dict()

        self.type_num = 0
        self.bigram_total = 0
        self.Precision = 0
        self.Recall_ratio = 0
        self.F_Measure = 0
        self.mark_Precision = 0

    # 将txt格式的语料库转换为csv格式，便于处理
    def txt2csv(self):
        print("-> txt2csv : div file ", self.div_txtpath, self.div_csvpath)
        list_div = []
        # 使用上下文管理器，离开后关闭文件
        with open(self.div_txtpath, "r", encoding='UTF-8') as f_div_txt:  # 打开文件
            for line in f_div_txt:
                list_div.append([item for item in line.split()])

        with open(self.div_csvpath, 'w', newline='', encoding='UTF-8') as f_div_csv:
            writer = csv.writer(f_div_csv)
            for row in list_div:
                if len(row) != 0:
                    writer.writerow(row)

        print("-> txt2csv : mark file ", self.mark_txtpath, self.mark_csvpath)
        list_mark = []
        # 使用上下文管理器，离开后关闭文件
        with open(self.mark_txtpath, "r", encoding='UTF-8') as f_mark_txt:  # 打开文件
            for line in f_mark_txt:
                list_mark.append([item for item in line.split()])

        with open(self.mark_csvpath, 'w', newline='', encoding='UTF-8') as f_mark_csv:
            writer = csv.writer(f_mark_csv)
            for row in list_mark:
                if len(row) != 0:
                    writer.writerow(row)

    # 读取数据
    def read_data(self):
        print("-> read_data : read div file")
        # 打开分词csv文件
        f_div = open(self.div_csvpath, "r", encoding='UTF-8')
        f_divreader = csv.reader(f_div)
        # 列表中元素为句子列表
        # 句子列表元素为词语和标点符号
        self.div_sentences = []
        for items in f_divreader:
            # 存储csv每一行中的词语
            temp_list = []
            for item in items:
                temp_list.append(item.strip())
            self.div_sentences.append(temp_list)
        f_div.close()

        print("-> read_data : read mark file")
        # 打开词性csv文件
        f_mark = open(self.mark_csvpath, "r", encoding='UTF-8')
        f_markreader = csv.reader(f_mark)
        # 列表中元素为句子列表
        # 句子列表元素为 词语/词性
        self.mark_sentences = []
        for items in f_markreader:
            # 存储csv每一行中的词语/词性
            temp_list = []
            for item in items:
                temp_list.append(item.strip())
            self.mark_sentences.append(temp_list)
        f_mark.close()

        print("-> read_data : count occurrences ")
        # 统计词性的出现次数
        # key:词性 ; value:出现次数
        self.marktype_cnt = dict()
        # 列表中的元素为句子列表
        for sentence in self.mark_sentences:
            for item in sentence:
                # 对/两侧进行切分，/后面为词性
                word_type = item.split('/')[-1]
                # 去掉标点符号
                word_type = word_type.split('!')[0].split(']')[0]
                # 如果该词性不在词典中，出现次数初始化为1，否则+1
                if word_type not in self.marktype_cnt:
                    self.marktype_cnt[word_type] = 1
                else:
                    self.marktype_cnt[word_type] += 1

        # 统计词性类别总数
        self.type_num = len(self.marktype_cnt)

        # print(self.div_sentences)
        # print(self.mark_sentences)
        # print(self.marktype_cnt)

    # 一元语法建模
    def unigram(self, sentences: list):
        print("-> unigram ")
        # unigram_dic存储连续两个词语及出现次数
        # key:word ; value:cnt
        self.unigram_dic = dict()
        # 起始与结束标记出现次数初始化为0
        self.unigram_dic['<BOS>'] = 0
        self.unigram_dic['<EOS>'] = 0
        for sentence in sentences:
            for word in sentence:
                # 对于每一个句子，<BOS><EOS>次数+1
                self.unigram_dic['<BOS>'] += 1
                self.unigram_dic['<EOS>'] += 1
                # 如果单词没在词典中，出现次数初始化为1
                if word not in self.unigram_dic:
                    self.unigram_dic[word] = 1
                # 如果单词在词典中，出现次数+1
                else:
                    self.unigram_dic[word] += 1
        # print(self.unigram_dic)

    # 二元语法建模
    def bigram(self, sentences: list):
        print("-> bigram ")
        # bigram_dic存储连续两个词语及出现次数
        # key:word_fir ; value:dic{ ket:word_sec ; valus:cnt }
        self.bigram_dic = dict()
        # 处理句子中第一个词语
        self.bigram_dic['<BOS>'] = dict()
        for sentence in sentences:

            # 第一个词语和最后一个词语需要同<BOS><EOS>单独处理
            # 如果第一个单词在<BOS>字典中没有出现，次数初始化为1，否则+1
            if sentence[0] not in self.bigram_dic['<BOS>']:
                self.bigram_dic['<BOS>'][sentence[0]] = 1
            else:
                self.bigram_dic['<BOS>'][sentence[0]] += 1

            # 从前向后遍历每一个词语
            for num in range(1, len(sentence) - 1):
                # 如果字典中没有word_fir
                if sentence[num] not in self.bigram_dic:
                    # 为word_fir建立字典
                    # word_sec出现次数初始化为0
                    self.bigram_dic[sentence[num]] = dict()
                    self.bigram_dic[sentence[num]][sentence[num + 1]] = 1
                # 字典中出现word_fir
                else:
                    # word_sec不在字典中，初始化出现次数为1
                    if sentence[num + 1] not in self.bigram_dic[sentence[num]]:
                        self.bigram_dic[sentence[num]][sentence[num + 1]] = 1
                    else:
                        self.bigram_dic[sentence[num]][sentence[num + 1]] += 1

            # 如果最后一个词语没在词典中
            # 建立词典，和<EOS>的出现次数初始化为1
            if sentence[len(sentence) - 1] not in self.bigram_dic:
                self.bigram_dic[sentence[len(sentence) - 1]] = dict()
                self.bigram_dic[sentence[len(sentence) - 1]]['<EOS>'] = 1
            # 如果最后一个词语在词典中
            else:
                # 如果<EOS>没在词典中，初始化出现次数为1，否则+1
                if '<EOS>' not in self.bigram_dic[sentence[len(sentence) - 1]]:
                    self.bigram_dic[sentence[len(sentence) - 1]]['<EOS>'] = 1
                else:
                    self.bigram_dic[sentence[len(sentence) - 1]]['<EOS>'] += 1

        self.bigram_total = 0
        for item in self.bigram_dic.keys():
            self.bigram_total += sum(self.bigram_dic[item].values())
        # print(self.bigram_total)
        # print(self.bigram_dic)

    # 生成viterbi算法所需的词性转移矩阵
    def gen_type_transfer(self, sentences: list):
        print("-> gen_type_transfer : generate transfer matrix")
        # key:type_fir ; value:{ key:type_sec ; value:cnt}
        self.type_transfer = dict()
        # 列表中的元素为句子列表
        for senternce in sentences:
            # 句子列表中第一个元素为日趋
            for num in range(1, len(senternce) - 1):
                # 提取前后两个词语的词性
                type_fir = senternce[num].split('/')[-1].split('!')[0].split(']')[0]
                type_sec = senternce[num + 1].split('/')[-1].split('!')[0].split(']')[0]
                # 如果第一个词语的词性不在字典中，为其建立字典，并初始化次数为1
                if type_fir not in self.type_transfer:
                    self.type_transfer[type_fir] = dict()
                    self.type_transfer[type_fir][type_sec] = 1
                else:
                    # 如果第二个词语的词性不在字典中，初始化出现次数为1，否则+1
                    if type_sec not in self.type_transfer[type_fir]:
                        self.type_transfer[type_fir][type_sec] = 1
                    else:
                        self.type_transfer[type_fir][type_sec] += 1
        # print(self.type_transfer)

    # 生成词性频度表
    def gen_type_P(self, sentences: list):
        print("-> gen_type_P : generate probability ")
        # 用字典存储词性概率频度
        # key:word ; value:{ key:word_type ; value:cnt}
        self.type_P = dict()
        # 依次遍历列表中每一个句子
        for sentence in sentences:
            # 句子列表中第一个元素为日期标签，跳过
            for num in range(1, len(sentence)):
                item = sentence[num]
                # /前面为词语，后面为词性
                word = item.split('/')[0].split('[')[0].split('{')[0]
                word_type = item.split('/')[-1].split('!')[0].split(']')[0]
                # 如果字典中没有该词语，则建立字典，初始化出现次数为1
                if word not in self.type_P:
                    self.type_P[word] = dict()
                    self.type_P[word][word_type] = 1
                else:
                    # 如果该词性没有在字典中出现，初始化出现次数为1，否则+1
                    if word_type not in self.type_P[word]:
                        self.type_P[word][word_type] = 1
                    else:
                        self.type_P[word][word_type] += 1
        # print(self.type_P)

    # 将句子列表转换为下标对应的集合，便于计算
    def list2set(self, sentence_list: list):
        sentence_set = set()
        # 下标从1开始
        num = 1
        # 对于列表中每一个词语，在集合添加(起始位置，终止位置)元组
        for word in sentence_list:
            sentence_set.add(tuple([num, num + len(word) - 1]))
            num += len(word)
        return sentence_set

    # 使用加一法处理未在词典中的词语
    # 返回P(word_sec|word_fir)
    def Additive_smoothing(self, bigram: dict, word_fir, word_sec):
        # print("-> Additive_smoothing :   ")
        P = 0
        # 如果还没有计算二元语法元素总数，报错
        if self.bigram_total == 0:
            print("----error!----bigram_total == 0----")
            return P
        # 如果字典中没有word_fir，出现次数为1
        if word_fir not in bigram:
            P = 1 / self.bigram_total
        else:
            # 如果字典中没有word_sec，出现次数为1，否则出现次数+1
            if word_sec not in bigram[word_fir]:
                P = 1 / (sum(bigram[word_fir].values()) + self.bigram_total)
            else:
                P = (bigram[word_fir][word_sec] + 1) / (sum(bigram[word_fir].values()) + self.bigram_total)
        return P

    # 生成词网
    def gen_net(self, unigram, sentencce):
        # print("-> gen_net : generate word net ",sentencce)
        # 列表中第i个元素为第i个字起始的词语
        # 词网第i行存储句子中第i个字开始的词语(序号从1开始）
        word_net = []
        len_sen = len(sentencce)
        # 生成len+2个空列表，为<BOS><EOS>准备
        for num in range(len_sen + 2):
            word_net.append([])
        # 从前向后遍历每一个词语
        for num in range(len_sen + 2):
            for j in range(num + 1, len_sen + 1):
                item = sentencce[num:j]
                # 如果一元语法中有这个词语，在词网中添加该词语
                if item in unigram:
                    word_net[num + 1].append(item)

        # 从前向后遍历词网中每一个元素
        num = 1
        while num < (len_sen - 1):
            # 发现空列表
            if not word_net[num]:
                # 向后依次遍历每一个列表，直到发现非空列表
                for j in range(num + 1, len_sen - 1):
                    if word_net[j]:
                        # 在空列表中添加未识别的词语，转至非空列表处继续循环
                        word_net[num].append(sentencce[num - 1:j - 1])
                        num = j
                        break
            # 列表不为空,跳转至本行最长词语末尾继续循环
            else:
                num += len(word_net[num][-1])
        word_net[len_sen + 1].append('<EOS>')

        # print(word_net)
        return word_net

    # 使用viterbi算法，基于统计方法进行分词
    def divide(self, sentence):
        # 根据句子生成词网
        word_net = self.gen_net(self.unigram_dic, sentence)
        word_net_len = len(word_net)
        # print("-> divide :  ",sentence,'  ',end='')
        # probability列表存储最大概率
        # id列表存储前一个词语的起始位置，便于回溯
        # temp_list列表存储相邻词语，便于回溯
        # backtrack为回溯结果列表
        probability = []
        id = []
        temp_list = []
        backtrack = []

        # 通过字典存储概率，词语起始位置，相邻词语的信息
        for num in range(word_net_len):
            probability.append(dict())
            id.append(dict())
            temp_list.append(dict())
        # 使用viterbi算法进行分词，依次遍历词网中的每一行
        for num in range(word_net_len - 1):
            # 初始化最大概率，词语序号
            if num == 0:
                for word in word_net[1]:
                    # 词网第一行存储句子中第一个可能的词语，该词语的前一个词为<BOS>
                    probability[1][word] = self.Additive_smoothing(self.bigram_dic, '<BOS>', word)
                    id[1][word] = 0
                    temp_list[1][word] = '<BOS>'
            else:
                # 句子中下标为num开始的词语为word_fir,num+len(word_fir)开始的词语为word_sec
                for word_fir in word_net[num]:
                    for word_sec in word_net[num + len(word_fir)]:
                        if word_fir in probability[num]:
                            # 如果word_sec在字典中，更新最大的概率，否则初始化最大概率
                            if word_sec in probability[num + len(word_fir)]:
                                # 计算当前状态的概率，和最大概率比较
                                P_max = probability[num + len(word_fir)][word_sec]
                                P_new = probability[num][word_fir] * self.Additive_smoothing(self.bigram_dic, word_fir,
                                                                                             word_sec)
                                if P_max < P_new:
                                    probability[num + len(word_fir)][word_sec] = P_new
                                    id[num + len(word_fir)][word_sec] = num
                                    temp_list[num + len(word_fir)][word_sec] = word_fir
                            # word_sec没在字典中，初始化当前概率为最大概率
                            else:
                                P_max = probability[num][word_fir] * self.Additive_smoothing(self.bigram_dic, word_fir,
                                                                                             word_sec)
                                probability[num + len(word_fir)][word_sec] = P_max
                                id[num + len(word_fir)][word_sec] = num
                                temp_list[num + len(word_fir)][word_sec] = word_fir
        # current_id=id[word_net_len-1]['<EOS>']
        # word=temp_list[word_net_len-1]['<EOS>']
        # backtrack.append(word)
        # 所有词语均已计算完成，回溯，得到最终结果
        word = '<EOS>'
        current_id = word_net_len - 1
        # backtrack.append(word)
        while True:
            # 在temp_list中查找前一个词语，更新current_id
            word_old = word
            word = temp_list[current_id][word_old]
            current_id = id[current_id][word_old]
            # 前一个词语的序号为0，到达<BOS>，停止回溯
            if current_id == 0:
                break
            else:
                backtrack.append(word)
        # 列表逆序排列即为分词结果
        backtrack.reverse()
        # print(backtrack)
        return backtrack

    # 使用viterbi算法进行词性标注
    def mark(self, sentence_list):
        # print("-> mark :  ",sentence_list,'  ',end='')
        # word_mark_dic中的元素为字典，存储单词的词性出现次数
        # probability列表存储最大概率
        # id列表存储前一个词语的词性，便于回溯
        # backtrack为回溯结果列表
        word_mark_dic = []
        probability = []
        id = []
        backtrack = []
        len_sentence = len(sentence_list)

        # 通过字典存储概率，词语词性
        for num in range(len_sentence):
            probability.append(dict())
            id.append(dict())
        # 从前向后遍历分词结果的每一个词语，在列表中存储词性和频度信息
        for word in sentence_list:
            # 只对字典中现有的词语进行词性标注，不在字典报错
            if word not in self.type_P:
                print("----mark error!----", word)
                return
            word_mark_dic.append(self.type_P[word])

        # 初始化viterbi算法的起始值
        # 第一个词语的所有词性概率均设置为1
        for word_type in word_mark_dic[0].keys():
            probability[0][word_type] = 1

        # 从前向后遍历所有词语，计算概率
        for num in range(len_sentence - 1):
            # 两个列表存储word_fir和word_sec可能的词性
            word_fir_types = list(word_mark_dic[num].keys())
            word_sec_types = list(word_mark_dic[num + 1].keys())
            # 计算delta的数值
            for type_sec in word_sec_types:
                cnt = 0
                P_max = -1
                cnt_max = 0
                for type_fir in word_fir_types:
                    # 如果word_sec的词性没在转移矩阵中出现，使用加一法计算概率
                    if type_sec not in self.type_transfer[type_fir]:
                        P_cur = probability[num][type_fir] * (1 / (self.marktype_cnt[type_fir] + self.type_num)) * \
                                (word_mark_dic[num + 1][type_sec] / self.marktype_cnt[type_sec])
                    else:
                        P_cur = probability[num][type_fir] * ((self.type_transfer[type_fir][type_sec] + 1) / (
                                self.marktype_cnt[type_fir] + self.type_num)) * \
                                (word_mark_dic[num + 1][type_sec] / self.marktype_cnt[type_sec])
                    # 记录最大的概率和对应的序号
                    if P_cur > P_max:
                        P_max = P_cur
                        cnt_max = cnt
                    cnt += 1
                # 向type_sec转移的概率设置为最大的概率，id设置为对应的word_fir词性
                probability[num + 1][type_sec] = P_max
                id[num + 1][type_sec] = word_fir_types[cnt_max]

        # 到达最后一个词，寻找最佳词性标记type_final，开始回溯
        type_list = list(probability[len_sentence - 1].values())
        P_list = list(probability[len_sentence - 1].keys())
        type_final = P_list[type_list.index(max(type_list))]
        backtrack.append(type_final)
        # 向后依次遍历每一个词，寻找最佳的词性标记
        for num in range(len_sentence - 1, 0, -1):
            type_old = type_final
            type_final = id[num][type_old]
            backtrack.append(type_final)
        # 将回溯结果逆序输出即为最终结果
        backtrack.reverse()
        # print(backtrack)
        return backtrack

    # 对分词及词性标注结果进行评价
    def evaluate(self, test_sentence: list, div_ground_truth: list, mark_ground_truth: list):
        print("-> evaluate :  ")
        #初始化正确率、召回率、F1值，词性标注正确率，分词次数
        self.Precision = 0
        self.Recall_ratio = 0
        self.mark_Precision = 0
        self.F_Measure = 0
        mark_num = 0
        len_test = len(test_sentence)

        #从前向后遍历测试数据
        for num in range(len_test):
            #生成分词的列表和词语对应序号的集合，便于计算评价指标
            test_div_list = self.divide(test_sentence[num])
            test_div_set = self.list2set(test_div_list)
            div_gt_set = self.list2set(div_ground_truth[num])
            #通过对集合求交集来计算正确率和召回率
            TP = test_div_set & div_gt_set
            p = len(TP) / len(test_div_set)
            r = len(TP) / len(div_gt_set)
            self.Precision += p
            self.Recall_ratio += r

            #仅当分词正确才进行词性标注
            if test_div_set == div_gt_set:
                #mark_num为词性标注的次数
                mark_num += 1
                test_mark_list = self.mark(test_div_list)
                mark_gt_list = mark_ground_truth[num]
                #cnt_a为词性标注过程中正确标注的次数
                cnt_a = 0
                len_mark_list = len(test_mark_list)
                for i in range(len_mark_list):
                    if test_mark_list[i] == mark_gt_list[i]:
                        cnt_a += 1
                #更新词性标注正确值的和
                self.mark_Precision += (cnt_a / len(test_mark_list))
            #打印当前进度
            if num % 2000 == 0:
                print("-> evaluate :  {:.2%}".format(num / len_test))
        #计算正确率、召回率、词性标注正确率的平均值，计算F1值
        self.Precision /= len_test
        self.Recall_ratio /= len_test
        self.F_Measure = 2 * self.Precision * self.Recall_ratio / (self.Precision + self.Recall_ratio)
        self.mark_Precision /= mark_num
        print("-> evaluate :  Precision{:.3f}  mark_Precision{:.3f}  Recall_ratio{:.3f}  F_Measure{:.3f}  "
              .format(self.Precision, self.mark_Precision, self.Recall_ratio, self.F_Measure))

    #初始化，生成语言模型
    def init_run(self):
        self.txt2csv()
        self.read_data()
        self.gen_type_transfer(self.mark_sentences)
        self.gen_type_P(self.mark_sentences)
        self.unigram(self.div_sentences)
        self.bigram(self.div_sentences)

