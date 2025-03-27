#ifndef RECTIFIED_LINEAR_H
#define RECTIFIED_LINEAR_H

#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include <cmath>
#include <utility>

constexpr auto NUM_ELEMS = 2;
constexpr auto BITS_PER_BYTE = 8;
using input_t = double;

template <std::size_t elems>
using pkt = ap_axiu<sizeof(input_t) * BITS_PER_BYTE * elems, 0, 0, 0>;
using pair = pkt<2>;

#endif
