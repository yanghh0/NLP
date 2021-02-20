//
//  CRF++ -- Yet Another CRF toolkit
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//

#ifndef CRFPP_FREELIST_H_
#define CRFPP_FREELIST_H_

#include <vector>
#include <cstring>

namespace CRFPP 
{

template <class T>
class Length 
{
public:
    size_t operator()(const T *str) const { return 1; }
};

class charLength 
{
public:
    size_t operator()(const char *str) const { return strlen(str) + 1; }
};

template <class T, class LengthFunc = Length<T> >
class FreeList 
{
private:
    std::vector<T *> freeList;
    size_t pi;      // 一个 size 空间被用掉多少
    size_t li;      // vector 列表索引
    size_t size;    // 列表内每个指针指向的空间大小为 size 字节

public:
    explicit FreeList(size_t _size): pi(0), li(0), size(_size) {}
    explicit FreeList(): pi(0), li(0), size(0) {}

    virtual ~FreeList() 
    {
        for (li = 0; li < freeList.size(); ++li) {
            delete [] freeList[li];
        }
    }

public:
    void free() { li = pi = 0; }

    T* alloc(size_t len = 1) 
    {
        if ((pi + len) >= size) {   // 当前 size 空间剩余部分不够存储 len 长度的字符串
            li++;                   // 移到下一个 size 空间
            pi = 0;                 // 新的空间，被用掉的为 0
        }
        if (li == freeList.size()) {
            freeList.push_back(new T[size]);
        }
        T* r = freeList[li] + pi;   // 当前指针位置
        pi += len;                  // 消耗掉 len 字节空间
        return r;
    }

    T* dup(T *src, size_t len = 0) 
    {
        if (!len) 
            len = LengthFunc()(src);
        T *p = alloc(len);
        if (src == 0) 
            memset(p, 0, len * sizeof (T));
        else        
            memcpy(p, src, len * sizeof(T));
        return p;
    }

    void set_size(size_t n) { size = n; }
};

typedef FreeList<char, charLength> StrFreeList;

}

#endif
