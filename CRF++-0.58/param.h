//
//  CRF++ -- Yet Another CRF toolkit
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//

#ifndef CRFPP_PARAM_H__
#define CRFPP_PARAM_H__

#include <map>
#include <string>
#include <vector>
#include <sstream>
#include "scoped_ptr.h"
#include "common.h"

namespace CRFPP 
{
namespace 
{

// 数据类型从 Source 转换到 Target，即实现任意数据类型转换
template <class Target, class Source>
Target lexical_cast(Source arg) 
{
    std::stringstream interpreter;
    Target result;
    if (!(interpreter << arg) || !(interpreter >> result) || !(interpreter >> std::ws).eof()) {
        scoped_ptr<Target> r(new Target());  // return default value
        return *r;
    }
    return result;
}

template <>
std::string lexical_cast<std::string, std::string>(std::string arg) 
{
    return arg;
}

}  // namespace

struct Option 
{
    const char *name;                // 选项完整的名字
    char        short_name;          // 选项名字缩写
    const char *default_value;       // 默认值
    const char *arg_description;     // 默认值类型
    const char *description;         // 选项说明
};

class Param 
{
private:
    std::map<std::string, std::string> conf_;
    std::vector<std::string>           rest_;
    std::string                        system_name_;
    std::string                        help_;
    std::string                        version_;
    whatlog                            what_;

public:
    bool open(int argc,  char **argv, const Option *opt);
    bool open(const char *arg,  const Option *opt);
    bool load(const char *filename);
    void clear();
    
    const std::vector<std::string>& rest_args() const { return rest_; }
    const char* program_name() const { return system_name_.c_str(); }
    const char *what() { return what_.str(); }
    const char* help() const { return help_.c_str(); }
    const char* version() const { return version_.c_str(); }
    int help_version() const;

    template <class T>
    T get(const char *key) const 
    {
        std::map<std::string, std::string>::const_iterator it = conf_.find(key);
        if (it == conf_.end()) {
            scoped_ptr<T> r(new T());
            return *r;
        }
        return lexical_cast<T, std::string>(it->second);
    }

    template <class T>
    void set(const char* key, const T &value, bool rewrite = true) 
    {
        std::string key2 = std::string(key);
        if (rewrite || (!rewrite && conf_.find(key2) == conf_.end()))
            conf_[key2] = lexical_cast<std::string, T>(value);
    }

    void dump_config(std::ostream *os) const;

    explicit Param() {}
    virtual ~Param() {}
};

}

#endif
