这个文件夹是conll_2003的简化版，就是数据集去掉了一些特征，只保留字符序列和标签序列。
处理代码为：
for file_type in ['train', 'dev', 'test']:
    raw_file_path = os.path.join("CoNLL-2003", file_type + '.txt')
    out_file_path = os.path.join("conll_2003_simple", file_type + '.bmes')
    content = ''
    with open(raw_file_path, 'r') as f:
        lines = []
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                lines.append(line)
            else:
                temp = line.split(' ')
                lines.append(' '.join([temp[0], temp[-1]]))
        content = '\n'.join(lines)
    with open(out_file_path, 'w') as f:
        f.write(content)   
    print(raw_file_path, out_file_path)

glove.6B.100d.txt 是预训练的词向量，需要先下载到这个文件夹下
!wget http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip