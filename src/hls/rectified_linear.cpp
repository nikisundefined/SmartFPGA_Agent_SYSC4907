#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include <cmath>
#include <utility>

// Converts an integer to a floating point and vice versa
// Used to convert the input from a stream
union fp_int {
    int64_t i;
    double fp;
};

struct output_t {
    fp_int value_a;
    fp_int value_b;
    fp_int value_c;
    fp_int value_d;
};

enum FPGAFunction : uint8_t {
    RectifiedLinear_GainBias,
    RectifiedLinear_MaxRatesIntercepts,
    RectifiedLinear_Step
};

// Type of a packet sent and received by a stream
typedef ap_axis<sizeof(fp_int) * 8,1,1,1> inputPkt; // Input 1 64-bit floating point number
typedef ap_axis<sizeof(output_t) * 8,1,1,1> outputPkt; // Output upto 4 64-bit floating point numbers

namespace RectifiedLinear {

    std::pair<double, double> gain_bias(const double &max_rates, const double &intercepts) {
        auto gain = max_rates / (1 - intercepts);
        auto bias = -intercepts * gain;
        return {gain, bias};
    }

    std::pair<double, double> max_rates_intercepts(const double &gain, const double &bias) {
        auto intercepts = -bias / gain;
        auto max_rates = gain * (1 - intercepts);
        return {max_rates, intercepts};
    }

    void step(const double &amplitude, const double &J, double &output) {
        output = amplitude * std::fmax(0.0, J);
    }

}

void nengofpga(uint8_t funcSelect, double amplitude,
    hls::stream<inputPkt> &A, hls::stream<inputPkt> &B, hls::stream<inputPkt> &C, hls::stream<outputPkt> &D) {
#pragma HLS INTERFACE s_axilite port=funcSelect
#pragma HLS INTERFACE s_axilite port=amplitude
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE axis port=A,B,C,D

    // Data structures used in parsing input streams
    fp_int a_data, b_data, c_data;
    output_t d_data;
    inputPkt a_pkt, b_pkt, c_pkt;
    // Load packet from input streams (non-blocking to allow fewer arguments to be sent at a time) [I think]
    A.read_nb(a_pkt);
    B.read_nb(b_pkt);
    C.read_nb(c_pkt);
    // Load data from packets
    a_data.i = a_pkt.data;
    b_data.i = b_pkt.data;
    c_data.i = c_pkt.data;

    // Prepare the output packet
    outputPkt d_pkt;
    d_pkt.dest = a_pkt.dest;
    d_pkt.id = a_pkt.id;
    d_pkt.keep = a_pkt.keep;
    d_pkt.last = a_pkt.last;
    d_pkt.strb = a_pkt.strb;
    d_pkt.user = a_pkt.user;

    std::pair<double, double> tmp;
    // Select the function needed
    switch (static_cast<FPGAFunction>(funcSelect)) {
        case RectifiedLinear_Step:
            RectifiedLinear::step(amplitude, a_data.fp, d_data.value_a.fp);
            break;
        case RectifiedLinear_GainBias:
            tmp = RectifiedLinear::gain_bias(a_data.fp, b_data.fp);
            d_data.value_a.fp = tmp.first;
            d_data.value_b.fp = tmp.second;
            break;
        case RectifiedLinear_MaxRatesIntercepts:
            tmp = RectifiedLinear::max_rates_intercepts(a_data.fp, b_data.fp);
            d_data.value_a.fp = tmp.first;
            d_data.value_b.fp = tmp.second;
            break;
    }
    // Move the data into the output packet
    d_pkt.data = reinterpret_cast<ap_int<sizeof(output_t) * 8> &>(d_data);
    // Send the packet back to the PS through the output stream
    D.write(d_pkt);
}


/* Old Working Code
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include <cmath>

union fp_int {
    int64_t i;
    double fp;
};

static_assert(sizeof(fp_int) * 8 == 64, "Incorrect size of int");
typedef ap_axis<sizeof(fp_int) * 8,1,1,1> transPkt;

void step(hls::stream< transPkt > &S_J, hls::stream< transPkt > &S_AMPLITUDE,
         hls::stream< transPkt > &S_OUTPUT) {
#pragma hls interface s_axilite port=return
#pragma HLS INTERFACE axis port=S_J,S_AMPLITUDE,S_OUTPUT

    fp_int amplitude, j;
    transPkt pkt_amplitude, pkt_j;
    while(1) {
        // Ensure streams still have data to process
        if (S_J.empty() or S_AMPLITUDE.empty()) break;
        // Read the values from them
        S_J.read(pkt_j);
        S_AMPLITUDE.read(pkt_amplitude);

        // Put them into the union to convert to double
        j.i = pkt_j.data;
        amplitude.i = pkt_amplitude.data;

        // Perform the step operation
        j.fp = std::fmax(0.0, j.fp) * amplitude.fp;

        // Write the result back into the amplitude packet and return the result
        pkt_amplitude.data = j.i;
        S_OUTPUT.write(pkt_amplitude);
        if (pkt_j.last or pkt_amplitude.last) {
            break;
        }
    }
}
*/