//
//  CRF++ -- Yet Another CRF toolkit
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//

#include <fstream>
#include <cstdio>
#include "param.h"
#include "common.h"

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

namespace CRFPP 
{
namespace 
{

void init_param(std::string *help, std::string *version, const std::string &system_name, const Option *opts) 
{
    *help = std::string(COPYRIGHT) + "\nUsage: " + system_name + " [options] files\n";
    *version = std::string(PACKAGE) + " of " + VERSION + '\n';

    size_t max = 0;

    for (size_t i = 0; opts[i].name; ++i) {
        size_t l = 1 + std::strlen(opts[i].name);             // opts[i].name=
        if (opts[i].arg_description)
            l += (1 + std::strlen(opts[i].arg_description));  // opts[i].arg_description + 一个空格
        max = std::max(l, max);
    }

    /*
    help = "CRF++: Yet Another CRF Tool Kit"
           "Copyright (C) 2005-2013 Taku Kudo, All rights reserved."
           "Usage: system_name [options] files"
           "-f, --freq=INT    use features that occuer no less than INT(default 1)"
           "-m, --maxiter=INT set INT for max iterations in LBFGS routine(default 10k)"
           "-c, --cost=FLOAT  set FLOAT for cost parameter(default 1.0)"
           "..."
    */
    for (size_t i = 0; opts[i].name; ++i) {
        size_t l = std::strlen(opts[i].name);
        if (opts[i].arg_description)
            l += (1 + std::strlen(opts[i].arg_description));  // =opts[i].arg_description
        *help += " -";
        *help += opts[i].short_name;
        *help += ", --";
        *help += opts[i].name;
        if (opts[i].arg_description) {
            *help += '=';
            *help += opts[i].arg_description;
        }
        for (; l <= max; l++) 
            *help += ' ';     // 补空格进行对齐
        *help += opts[i].description;
        *help += '\n';
    }

    *help += '\n';
    return;
}

}  // namespace

void Param::dump_config(std::ostream *os) const 
{
    for (std::map<std::string, std::string>::const_iterator it = conf_.begin(); it != conf_.end(); ++it) {
        *os << it->first << ": " << it->second << std::endl;
    }
}

bool Param::load(const char *filename) 
{
    std::ifstream ifs(WPATH(filename));
    CHECK_FALSE(ifs) << "no such file or directory: " << filename;

    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.size() || (line.size() && (line[0] == ';' || line[0] == '#'))) continue;

        size_t pos = line.find('=');
        CHECK_FALSE(pos != std::string::npos) << "format error: " << line;

        size_t s1, s2;
        for (s1 = pos + 1; s1 < line.size() && isspace(line[s1]); s1++);           // 跳过空格
        for (s2 = pos - 1; static_cast<long>(s2) >= 0 && isspace(line[s2]); s2--); // 跳过空格
        const std::string value = line.substr(s1, line.size() - s1);
        const std::string key   = line.substr(0, s2 + 1);
        set<std::string>(key.c_str(), value, false);
    }
    return true;
}

bool Param::open(int argc, char **argv, const Option *opts) 
{
    /*
    e.g.
        ../../crf_learn -c 10.0 template train.data model
        argc = 6
        argv = ["../../crf_learn",
                "-c",
                "10.0",
                "template",
                "train.data",
                "model"]
    */
    int ind = 0;
    int _errno = 0;

#define GOTO_FATAL_ERROR(n) {                   \
    _errno = n;                                 \
    goto FATAL_ERROR; } while (0)

    if (argc <= 0) {
        system_name_ = "unknown";
        return true;  // this is not error
    }

    system_name_ = std::string(argv[0]);
    init_param(&help_, &version_, system_name_, opts);

    // 存在默认值的选项，先在字典里生成 pair
    for (size_t i = 0; opts[i].name; ++i) {
        if (opts[i].default_value) 
            set<std::string>(opts[i].name, opts[i].default_value);
    }

    for (ind = 1; ind < argc; ind++) {
        if (argv[ind][0] == '-') {
            // long options: e.g. --maxiter=100000
            if (argv[ind][1] == '-') {
                char *s;
                // 扫描 name 部分
                for (s = &argv[ind][2]; *s != '\0' && *s != '='; s++);
                    size_t len = (size_t)(s - &argv[ind][2]);
                if (len == 0) return true;  // stop the scanning

                bool hit = false;
                size_t i = 0;
                for (i = 0; opts[i].name; ++i) {
                    size_t nlen = std::strlen(opts[i].name);
                    if (nlen == len && std::strncmp(&argv[ind][2], opts[i].name, len) == 0) {
                        hit = true;
                        break;
                    }
                }
                if (!hit) GOTO_FATAL_ERROR(0);
                if (opts[i].arg_description) {
                    if (*s == '=') {
                        if (*(s + 1) == '\0') GOTO_FATAL_ERROR(1);
                        set<std::string>(opts[i].name, s + 1);     // 提供了参数值则更新字典
                    } else {
                        if (argc == (ind + 1)) GOTO_FATAL_ERROR(1);
                        set<std::string>(opts[i].name, argv[++ind]);
                    }
                } else {     // 选项无需提供值
                    if (*s == '=') GOTO_FATAL_ERROR(2);
                    set<int>(opts[i].name, 1);
                }
              // short options: e.g. -m 100000
            } else if (argv[ind][1] != '\0') {
                size_t i = 0;
                bool hit = false;
                for (i = 0; opts[i].name; ++i) {
                    if (opts[i].short_name == argv[ind][1]) {
                        hit = true;
                        break;
                    }
                }
                if (!hit) GOTO_FATAL_ERROR(0);
                if (opts[i].arg_description) {
                    if (argv[ind][2] != '\0') {
                        set<std::string>(opts[i].name, &argv[ind][2]);
                    } else {
                        if (argc == (ind + 1)) GOTO_FATAL_ERROR(1);
                        set<std::string>(opts[i].name, argv[++ind]);
                    }
                } else {
                    if (argv[ind][2] != '\0') GOTO_FATAL_ERROR(2);
                    set<int>(opts[i].name, 1);
                }
            }
        } else {
            rest_.push_back(std::string(argv[ind]));  // others，如 template train.data model
        }
    }

    return true;

FATAL_ERROR:
    switch (_errno) {
        case 0: WHAT << "unrecognized option `" << argv[ind] << "`"; break;
        case 1: WHAT << "`" << argv[ind] << "` requires an argument";  break;
        case 2: WHAT << "`" << argv[ind] << "` doesn't allow an argument"; break;
    }
    return false;
}

void Param::clear() 
{
    conf_.clear();
    rest_.clear();
}

bool Param::open(const char *arg, const Option *opts) 
{
    char str[BUF_SIZE];
    std::strncpy(str, arg, sizeof(str));
    char* ptr[64];
    unsigned int size = 1;
    ptr[0] = const_cast<char*>(PACKAGE);

    for (char *p = str; *p;) {
        while (isspace(*p)) *p++ = '\0';   // 如果是空格，则指针右移
        if (*p == '\0') 
            break;
        ptr[size++] = p;                   // 把指向当前字符串起始位置的指针保存
        if (size == sizeof(ptr)) break;    // 存储超过64个参数，则溢出
        while (*p && !isspace(*p)) p++;    // 如果是字符，则指针右移
    }
    return open(size, ptr, opts);
}

int Param::help_version() const 
{
    if (get<bool>("help")) {
        std::cout << help();
        return 0;
    }

    if (get<bool>("version")) {
        std::cout << version();
        return 0;
    }
    return 1;
}

} // CRFPP
