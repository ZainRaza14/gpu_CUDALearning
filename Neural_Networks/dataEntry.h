#ifndef _DATAENTRY
#define _DATAENTRY

#include <iostream>
#include <vector>

using namespace std;

class dataEntry
{
public:
    float* pattern;
    float* target;

    dataEntry(float* p, float* t) : pattern(p), target(t) {}

    ~dataEntry()
    {
        delete[] pattern;
        delete[] target;
    }
};

#endif
