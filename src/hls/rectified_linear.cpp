#include "rectified_linear.h"
#define ENABLE_MAX_RATES_INTERCEPTS
#define ENABLE_GAIN_BIAS
#define ENABLE_STEP

#ifdef ENABLE_GAIN_BIAS

std::pair<double, double> _gain_bias(const double &max_rates, const double &intercepts) {
	auto gain = max_rates / (1 - intercepts);
	auto bias = -intercepts * gain;
	return {gain, bias};
}

void gain_bias(hls::stream<pair> &A, hls::stream<pair> &B) {
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS INTERFACE axis port=A,B
	pair tmp;
	double *val_a;
	double *val_b;
	do {
		tmp = A.read();
		val_a = (double *)&tmp.data;
		val_b = ((double *)&tmp.data) + 1;

		auto gb = _gain_bias(*val_a, *val_b);
		*val_a = gb.first;
		*val_b = gb.second;

		B.write(tmp);
	} while (not tmp.last);
}

#endif

#ifdef ENABLE_MAX_RATES_INTERCEPTS

std::pair<double, double> _max_rates_intercepts(const double &gain, const double &bias) {
	auto intercepts = -bias / gain;
	auto max_rates = gain * (1 - intercepts);
	return {max_rates, intercepts};
}

void max_rates_intercepts(hls::stream<pair> &A, hls::stream<pair> &B) {
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS INTERFACE axis port=A,B
	pair tmp;
	double *val_a;
	double *val_b;
	do {
		tmp = A.read();
		val_a = (double *)&tmp.data;
		val_b = ((double *)&tmp.data) + 1;

		auto mri = _max_rates_intercepts(*val_a, *val_b);
		*val_a = mri.first;
		*val_b = mri.second;

		B.write(tmp);
	} while (not tmp.last);
}

#endif

#ifdef ENABLE_STEP

double _step(const double &amplitude, const double &J) {
	return amplitude * (J < 0 ? 0.0 : J);
}

void step(hls::stream<pair> &A, hls::stream<pair> &B) {
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS INTERFACE axis port=A,B
	pair tmp;
	double *val_a;
	double *val_b;
	do {
		tmp = A.read();
		val_a = (double *)&tmp.data;
		val_b = ((double *)&tmp.data) + 1;

		auto s = _step(*val_a, *val_b);
		*val_a = s;
		*val_b = 0.0;

		B.write(tmp);
	} while (not tmp.last);
}

#endif

void nengofpga(hls::stream<pkt<4>> &A, hls::stream<pkt<4>> &B) {
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS INTERFACE axis port=A,B

	pkt<4> tmp;
	uint64_t *val_a;
	double *val_b;
	double *val_c;

	std::pair<double, double> a;
	double b;

	do {
		tmp = A.read();

		val_a = (uint64_t *)&tmp.data;
		val_b = ((double *)&tmp.data) + 1;
		val_c = ((double *)&tmp.data) + 2;

		switch (*val_a) {
		case 0:
			a = _gain_bias(*val_b, *val_c);
			*val_b = a.first;
			*val_c = a.second;
			break;
		case 1:
			a = _max_rates_intercepts(*val_b, *val_c);
			*val_b = a.first;
			*val_c = a.second;
			break;
		case 2:
			b = _step(*val_b, *val_c);
			*val_b = b;
			*val_c = 0.0;
			break;
		default:
			break;
		}

		B.write(tmp);
	} while (not tmp.last);
}
