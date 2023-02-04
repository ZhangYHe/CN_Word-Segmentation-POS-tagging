# CN_word-segmentation-POS-tagging
基于python的中文文本分词和词性标注

## DivMark_CN使用说明

### 1.数据准备

DivMark_CN初始化参数为分词及词性标注的语料库文件，格式为.txt。

```python
    def __init__(self, div_txtpath, mark_txtpath):
        # 初始化语料数据文件路径
        self.div_txtpath = div_txtpath
        self.mark_txtpath = mark_txtpath
```

### 2.初始化

调用init_run函数，进行初始化。将txt文件转换为csv文件，便于处理。读取语料库的数据，生成词性转移矩阵、词性频度表，并进行一元语法、二元语法建模。

```python
#初始化，生成语言模型
    def init_run(self):
        self.txt2csv()
        self.read_data()
        self.gen_type_transfer(self.mark_sentences)
        self.gen_type_P(self.mark_sentences)
        self.unigram(self.div_sentences)
        self.bigram(self.div_sentences)
```

### 3.分词

调用divide函数，使用viterbi算法进行分词。函数参数为句子字符串，返回值为分词结果的列表。

```python
# 使用viterbi算法，基于统计方法进行分词
    def divide(self, sentence):
```

### 4.词性标注

调用mark函数，使用viterbi算法进行词性标注。函数参数为分词后的列表，返回值为词语词性的列表。

```python
# 使用viterbi算法进行词性标注
    def mark(self, sentence_list):
```

### 5.结果评价

调用evaluate函数，对分词及词性标注结果进行评价，计算分词正确率、词性标注正确率、召回率、F1值。函数参数为句子列表，正确分词结果列表，正确词性标注结果列表。

```python
# 对分词及词性标注结果进行评价
    def evaluate(self, test_sentence: list, div_ground_truth: list
    , mark_ground_truth: list):
```
