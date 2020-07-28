import os
from pyltp import Segmentor, Postagger, Parser
import re
import codecs
import json


class LtpParser:
    def __init__(self):
        LTP_DIR = "ltp_data_v3.4.0"
        self.segmentor = Segmentor()
        self.segmentor.load(os.path.join(LTP_DIR, "cws.model"))

        self.postagger = Postagger()
        self.postagger.load(os.path.join(LTP_DIR, "pos.model"))

        self.parser = Parser()
        self.parser.load(os.path.join(LTP_DIR, "parser.model"))


    '''句法分析---为句子中的每个词语维护一个保存句法依存儿子节点的字典'''
    def build_parse_child_dict(self, words, postags, arcs):
        child_dict_list = []
        format_parse_list = []
        for index in range(len(words)):
            child_dict = dict()
            for arc_index in range(len(arcs)):
                if arcs[arc_index].head == index+1:   #arcs的索引从1开始
                    if arcs[arc_index].relation in child_dict:
                        child_dict[arcs[arc_index].relation].append(arc_index)
                    else:
                        child_dict[arcs[arc_index].relation] = []
                        child_dict[arcs[arc_index].relation].append(arc_index)
            child_dict_list.append(child_dict)
        rely_id = [arc.head for arc in arcs]  # 提取依存父节点id
        relation = [arc.relation for arc in arcs]  # 提取依存关系
        heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]  # 匹配依存父节点词语
        for i in range(len(words)):
            # ['ATT', '李克强', 0, 'nh', '总理', 1, 'n']
            a = [relation[i], words[i], i, postags[i], heads[i], rely_id[i]-1, postags[rely_id[i]-1]]
            format_parse_list.append(a)

        return child_dict_list, format_parse_list

    '''parser主函数'''
    def parser_main(self, sentence):
        words = list(self.segmentor.segment(sentence))   # 分词
        postags = list(self.postagger.postag(words))     # 词性标注
        arcs = self.parser.parse(words, postags)         # 句法分析
        child_dict_list, format_parse_list = self.build_parse_child_dict(words, postags, arcs)

        return words, postags, child_dict_list, format_parse_list

# parse = LtpParser()
# # sentence = '李克强昨日访问了美国著名的一个庄园'
# sentence = '轮状病毒性肠炎可能会让你觉得陌生，但它有一个通俗的名字“秋季腹泻”'
# words, postags, child_dict_list,format_parse_list = parse.parser_main(sentence)
# print(words, len(words))
# print(postags, len(postags))
# print(child_dict_list, len(child_dict_list))
# print(format_parse_list, len(format_parse_list))
#

class TripleExtractor:
    def __init__(self):
        self.parser = LtpParser()

    '''文章分句处理, 切分长句，冒号，分号，感叹号等做切分标识'''
    def split_sents(self, content):
        return [sentence for sentence in re.split(r'[？?！!。；;：:\n\r]', content) if sentence]

    '''三元组抽取主函数'''
    def ruler2(self, words, postags, child_dict_list, arcs):
        svos = []
        verd_list = []
        for index in range(len(postags)):
           if postags[index] == 'v':
                if words[index] not in ['说', '电', '叫', '晚', '是', '为', '开', '说', '称', '报道', '摄', '摄影']:
                # 抽取以谓词为中心的事实三元组
                    child_dict = child_dict_list[index]

                    # 主谓宾
                    if 'SBV' in child_dict and 'VOB' in child_dict:
                        if 'DBL' in child_dict:
                            r = words[index]
                            e1, postfix = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                            e3, _ = self.complete_e(words, postags, child_dict_list, child_dict['DBL'][0])
                            e2, _ = self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                            svos.append([[e1, postfix], [r, 'pad'], [e3, e2]])
                            verd_list.append(r)
                        else:
                            r = words[index]
                            e1, postfix1 = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                            e2, postfix2 = self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                            svos.append([[e1, postfix1], [r, 'pad'], [e2, postfix2]])
                            verd_list.append(r)
                    # 主谓
                    elif 'SBV' in child_dict:
                        r = words[index]
                        e1, postfix = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                        if e1 not in ['我', '你', '他', '她', '我们', '你们', '他们', '她们', '这', '那', '这些', '那些']:
                            svos.append([[e1, postfix], [r, 'pad'], ['pad', 'pad']])
                            verd_list.append(r)
                    # 动宾关系，这个时候必须要有兼语（有-->警察自杀）
                    elif 'VOB' in child_dict:
                        if 'DBL' in child_dict:
                            r = words[index]
                            e1, _ = self.complete_e(words, postags, child_dict_list, child_dict['DBL'][0])
                            e2, _ = self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                            # event = [r, e1, e2]
                            # svos.append([r,e1, e2])
                            svos.append([[e1, 'pad'], [r, 'pad'], [e2, 'pad']])  # 三元组形式
                            verd_list.append(r)
                        # 谓语-宾语结构会多出很多无意义的事件
                        # else:
                        #     r = words[index]
                        #     e2 = self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                        #     svos.append([r, e2])
                    # # 定语后置，动宾关系
                    # relation = arcs[index][0]
                    # head = arcs[index][2]
                    # if relation == 'ATT':
                    #     if 'VOB' in child_dict:
                    #         e1 = self.complete_e(words, postags, child_dict_list, head - 1)
                    #         r = words[index]
                    #         e2 = self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                    #         temp_string = r + e2
                    #         if temp_string == e1[:len(temp_string)]:
                    #             e1 = e1[len(temp_string):]
                    #         if temp_string not in e1:
                    #             svos.append([e1, r, e2])
                    # 含有介宾关系的主谓动补关系（做--->完了作业）
                    elif 'SBV' in child_dict and 'CMP' in child_dict:
                        e1, _ = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                        cmp_index = child_dict['CMP'][0]
                        r1 = words[index]
                        r2 = words[cmp_index]
                        if 'POB' in child_dict_list[cmp_index]:
                            e2, postfix = self.complete_e(words, postags, child_dict_list, child_dict_list[cmp_index]['POB'][0])
                            svos.append([[e1, 'pad'], [r1, r2], [e2, postfix]])  # 这里谓词不止一个成分
                            verd_list.append(r)
        return svos, verd_list

    '''对找出的主语或者宾语进行扩展'''
    def complete_e(self, words, postags, child_dict_list, word_index):
        child_dict = child_dict_list[word_index]
        # prefix = ''
        # # 把定语扩充取消，这个事件表示就是抽象的事件表示
        # if 'ATT' in child_dict:
        #     for i in range(len(child_dict['ATT'])):
        #         prefix += self.complete_e(words, postags, child_dict_list, child_dict['ATT'][i])
        postfix = 'pad'
        if postags[word_index] == 'v':
            if 'VOB' in child_dict:
                postfix, _ = self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
            # if 'SBV' in child_dict:
            #     prefix = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0]) + prefix

        # return prefix + words[word_index] + postfix
        return  words[word_index], postfix   # 不要扩充定语，扩充动宾

    '''程序主控函数'''
    def triples_main(self, content):
        sentences = self.split_sents(content)
        svos = []
        verbs_list = []
        for sentence in sentences:
            words, postags, child_dict_list, arcs = self.parser.parser_main(sentence)
            # self.parser.release_model()
            svo, verb = self.ruler2(words, postags, child_dict_list, arcs)
            svos += svo
            verbs_list += verb
        return svos, verbs_list
    def release_model(self):
        self.parser.segmentor.release()
        self.parser.parser.release()
        self.parser.postagger.release()



def test():
    content8 = '今年以来，已有10名纽约警察局警察自杀，警察的心理健康状态备受关注。'
    extractor = TripleExtractor()
    svos = extractor.triples_main(content8)
    print('svos', svos)

# test()


def loadData(file):
    Data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            # line = line.split(":")[1]
            Data.append(line)
    return Data


if __name__ == '__main__':
    file = 'text_res.txt'
    extractor = TripleExtractor()
    datMat = loadData(file)
    datamet = []
    for text in datMat:
        if len(text)>2000:
            continue
        res = []
        id = text.split(':')[0]
        content = text.split(':')[1]
        content = content.replace('\n', '').replace('\t', '')
        data, verbs_list = extractor.triples_main(content)
        ## 这个是写谓词表的
        # verbs = ','.join(verbs_list)
        # f1 = codecs.open('verb.txt', 'a', 'utf-8')
        # f1.write(verbs+ '\n')
        data_ = []
        for triple in data:
            # event = ''.join(triple)  # 直接抽取三元组把
            data_.append(triple)
        if len(data_) < 2:
            continue
        triple_dict = {}
        triple_dict[id] = data_
        with open(os.path.join('event_triples/', id + '.json'), 'w', encoding='utf-8') as f:
            json.dump(triple_dict, f,ensure_ascii=False)
        # event_all = ','.join(data_)
        # res.append(str(id))
        # res.append(event_all)
        # res_str = ':'.join(res)
        # if len(res_str) < 20:
        #     continue
        # f = codecs.open('final.txt', 'a', 'utf-8')
        # f.write(res_str+'\n')



