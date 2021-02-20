//
//  CRF++ -- Yet Another CRF toolkit
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//

#ifndef CRFPP_ENCODER_H_
#define CRFPP_ENCODER_H_

#include "common.h"

namespace CRFPP 
{

class Encoder 
{
public:
    enum { CRF_L2, CRF_L1, MIRA };
    
    bool learn(const char *, 
    	       const char *,
    	       const char *, 
    	       bool, 
    	       size_t, 
    	       size_t, 
    	       double, 
    	       double, 
    	       unsigned short, 
    	       unsigned short, 
    	       int);
    bool convert(const char *text_file, const char *binary_file);
    const char* what() { return what_.str(); }

private:
    whatlog what_;
};

}  // CRFPP

#endif
