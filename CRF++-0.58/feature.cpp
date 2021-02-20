//
//  CRF++ -- Yet Another CRF toolkit
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//

#include "feature_index.h"
#include "common.h"
#include "node.h"
#include "path.h"
#include "tagger.h"

namespace CRFPP 
{

const size_t kMaxContextSize = 8;

// 如果某行缺失，则特征用下面符号来表示
const char *BOS[kMaxContextSize] = { "_B-1", "_B-2", "_B-3", "_B-4",
                                     "_B-5", "_B-6", "_B-7", "_B-8" };
const char *EOS[kMaxContextSize] = { "_B+1", "_B+2", "_B+3", "_B+4",
                                     "_B+5", "_B+6", "_B+7", "_B+8" };

const char *FeatureIndex::getIndex(const char *&p, size_t pos, const TaggerImpl &tagger) const 
{
    if (*p++ !='[') {
        return 0;
    }

    int col = 0;
    int row = 0;
    int neg = 1;

    if (*p++ == '-') {
        neg = -1;
    } else {
        --p;
    }

    for (; *p; ++p) {
        switch (*p) {
            case '0': 
            case '1': 
            case '2': 
            case '3': 
            case '4':
            case '5': 
            case '6': 
            case '7': 
            case '8': 
            case '9':
                row = 10 * row + (*p - '0');
                break;
            case ',':
                ++p;
                goto NEXT1;
            default: 
                return  0;
        }
    }

NEXT1:

    for (; *p; ++p) {
        switch (*p) {
            case '0': 
            case '1': 
            case '2': 
            case '3': 
            case '4':
            case '5': 
            case '6': 
            case '7': 
            case '8': 
            case '9':
                col = 10 * col + (*p - '0');
                break;
            case ']': 
                goto NEXT2;
            default: 
                return 0;
        }
    }

NEXT2:
    row *= neg;
    if (row < -static_cast<int>(kMaxContextSize) || row > static_cast<int>(kMaxContextSize) ||
        col < 0 || col >= static_cast<int>(tagger.xsize())) {
        return 0;
    }

    // TODO(taku): very dirty workaround
    if (check_max_xsize_) {
        max_xsize_ = std::max(max_xsize_, static_cast<unsigned int>(col + 1));
    }

    const int idx = pos + row;
    if (idx < 0) {
        return BOS[-idx-1];
    }

    if (idx >= static_cast<int>(tagger.size())) {
        return EOS[idx - tagger.size()];
    }

    return tagger.x(idx, col);
}

// p 是指向某一行模板字符串的指针，如: U00:%x[-2,0] 
bool FeatureIndex::applyRule(string_buffer *os, const char *p, size_t pos, const TaggerImpl& tagger) const 
{
    os->assign("");  // clear
    const char *r;

    for (; *p; p++) {
        switch (*p) {
            default:
                *os << *p;  // U00:
                break;
            case '%':
                switch (*++p) {
                    case 'x':
                        ++p;
                        r = getIndex(p, pos, tagger);  // [-2,0] 
                        if (!r) {
                            return false;
                        }
                        *os << r;
                        break;
                    default:
                        return false;
                }
                break;
        }
    }

    *os << '\0';
    return true;
}

void FeatureIndex::rebuildFeatures(TaggerImpl *tagger) const 
{
    size_t fid = tagger->feature_id();
    const size_t thread_id = tagger->thread_id();

    Allocator *allocator = tagger->allocator();
    allocator->clear_freelist(thread_id);
    FeatureCache *feature_cache = allocator->feature_cache();

    // 为每个词以及对应的所有可能的label，构造节点
    for (size_t cur = 0; cur < tagger->size(); ++cur) {  // 遍历一个句子的每个词
        const int *f = (*feature_cache)[fid++];          // 取出每个词的特征列表，词的特征列表对应特征模板里的Unigram特征
        for (size_t i = 0; i < y_.size(); ++i) {         //每个词都对应全部可能的label， 每个label用数组的下标表示
            Node *n = allocator->newNode(thread_id);
            n->clear();
            n->x = cur;          // 当前词
            n->y = i;            // 当前词的label
            n->fvector = f;      // 只要保存该词抽取出来的所有特征 id，通过特征的 id 自然能找到特征函数的 id                 
            tagger->set_node(n, cur, i);  // 用一个二维数组 node_ 存放每个节点，node_[cur][i] = n
        }
    }

    // 从第二个词开始构造节点之间的边，两个词之间有 y_.size() * y_.size() 条边
    for (size_t cur = 1; cur < tagger->size(); ++cur) {
        const int *f = (*feature_cache)[fid++];       // 取出每个边的特征列表，边的特征列表对应特征模板里的Bigram特征
        for (size_t j = 0; j < y_.size(); ++j) {
            for (size_t i = 0; i < y_.size(); ++i) {
                Path *p = allocator->newPath(thread_id);
                p->clear();
                p->add(tagger->node(cur - 1, j), tagger->node(cur, i));
                p->fvector = f;
            }
        }
    }
}

#define ADD { const int id = getID(os.c_str());         \
    if (id != -1) feature.push_back(id); } while (0)

bool FeatureIndex::buildFeatures(TaggerImpl *tagger) const 
{
    string_buffer os;
    std::vector<int> feature;

    FeatureCache *feature_cache = tagger->allocator()->feature_cache();
    tagger->set_feature_id(feature_cache->size());

    for (size_t cur = 0; cur < tagger->size(); ++cur) {   // 遍历一个句子的每个词，抽取每个词的状态特征
        for (std::vector<std::string>::const_iterator it = unigram_templs_.begin(); it != unigram_templs_.end(); ++it) {
            if (!applyRule(&os, it->c_str(), cur, *tagger)) {
                return false;
            }
            ADD; // 将根据特征 os，获取该特征对应特征函数的 id，如果不存在该特征函数，则生成新的 id，将该 id 添加到 feature 变量中
        }
        feature_cache->add(feature);
        feature.clear();
    }

    for (size_t cur = 1; cur < tagger->size(); ++cur) {   // 遍历一个句子的每个词，抽取每个词的转移特征
        for (std::vector<std::string>::const_iterator it = bigram_templs_.begin(); it != bigram_templs_.end(); ++it) {
            if (!applyRule(&os, it->c_str(), cur, *tagger)) {
                return false;
            }
            ADD;
        }
        feature_cache->add(feature);
        feature.clear();
    }

    return true;
}
#undef ADD
}
