#include "../nengo.h"

namespace nengo::neuron_types {

}
/*
pybind11::tuple RectifiedLinear::gain_bias(__attribute__((unused)) const pybind11::object &self,
                                                    const NeuronType::GenericNeuralVector &max_rates,
                                                    const NeuronType::GenericNeuralVector &intercepts) {
    // Extra checking done in generic implementation to prevent errors
    if (max_rates.rows() != intercepts.rows())
        throw std::invalid_argument("max_rates and intercepts must have the same shape");
    pybind11::print("Hello from RectifiedLinear.gain_bias in C++");

    // Create return vectors
    Eigen::VectorXd gain(max_rates.rows());
    Eigen::VectorXd bias(intercepts.rows());

    // Compute gain for each index
    for (auto i = 0; i != max_rates.size(); ++i) {
        gain(i) = max_rates(i) / (1 - intercepts(i));
    }

    // Compute bias for each index
    for (auto i = 0; i != intercepts.size(); ++i) {
        bias(i) = -intercepts(i) * gain(i);
    }

    return pybind11::make_tuple(gain, bias);
}

pybind11::tuple RectifiedLinear::max_rates_intercepts(__attribute__((unused)) const pybind11::object &self,
                                                               const NeuronType::GenericNeuralVector &gain,
                                                               const NeuronType::GenericNeuralVector &bias) {
    // Extra size checking to since input is generically sized
    if (gain.rows() != bias.rows())
        throw std::invalid_argument("gain and bias must have the same shape");

    // Create return vectors
    Eigen::VectorXd intercepts(gain.rows());
    Eigen::VectorXd max_rates(bias.rows());

    // Compute intercepts for each element
    for (auto i = 0; i != gain.size(); ++i) {
        intercepts(i) = -bias(i) / gain(i);
    }

    // Compute max_rates for each element
    for (auto i = 0; i < bias.size(); ++i) {
        max_rates(i) = gain(i) / (1 - intercepts(i));
    }
    return pybind11::make_tuple(max_rates, intercepts);
}

void RectifiedLinear::step(const pybind11::object &self,
                                    __attribute__((unused)) const float dt,
                                    const NeuronType::GenericNeuralMatrix &J,
                                    NeuronType::GenericNeuralMatrix &output) {
    if (J.rows() != output.rows() or J.cols() != output.cols())
        throw std::invalid_argument("J and output must have the same shape");
    const auto amplitude = self.attr("amplitude").cast<float>();
    for (auto i = 0; i < J.rows(); ++i) {
        for (auto j = 0; j < J.cols(); ++j) {
            output(i, j) = amplitude * std::max(J(i, j), 0.0);
        }
    }
}

Eigen::MatrixXd SpikingRectifiedLinear::rates(pybind11::object &self,
                                           const NeuronType::GenericNeuralMatrix &x,
                                           const NeuronType::GenericNeuralMatrix &gain,
                                           const NeuronType::GenericNeuralMatrix &bias) {
    const auto J = self.attr("current")(x, gain, bias).cast<NeuronType::GenericNeuralMatrix>();
    NeuronType::GenericNeuralMatrix out(J);
    RectifiedLinear::step(self, 1.0f, J, out);
    return out;
}

void SpikingRectifiedLinear::step(const pybind11::object &self, const float dt,
                                           const NeuronType::GenericNeuralMatrix &J,
                                           NeuronType::GenericNeuralMatrix &output,
                                           NeuronType::GenericNeuralMatrix &voltage) {
    const auto amplitude = self.attr("amplitude").cast<float>();
    for (auto i = 0; i < J.rows(); ++i) {
        for (auto j = 0; j < J.cols(); ++j) {
            const auto iter_voltage = voltage(i, j) + std::max(J(i, j), 0.0) * dt;
            const auto n_spikes = std::floor(J(i, j));
            output(i, j) = (amplitude / dt) * n_spikes;
            voltage(i, j) = iter_voltage - n_spikes;
        }
    }
}
*/
PYBIND11_MODULE(nengocpp, m) {

    pybind11::class_<nengo::neuron_types::NeuronType>(m, "NeuronType")
    .def(pybind11::init<>())
    .def("current", &nengo::neuron_types::NeuronType::current<Eigen::Dynamic, Eigen::Dynamic>);

    // pybind11::class_<RectifiedLinear>(m, "RectifiedLinearImpl")
    //         .def(pybind11::init<>())
    //         .def("gain_bias", &RectifiedLinear::gain_bias,
    //             pybind11::arg("max_rates"), pybind11::arg("intercepts"))
    //         .def("max_rates_intercepts", &RectifiedLinear::max_rates_intercepts,
    //             pybind11::arg("gain"), pybind11::arg("bias"))
    //         .def("step", &RectifiedLinear::step,
    //             pybind11::arg("dt"), pybind11::arg("J"), pybind11::arg("output"));
    //
    // pybind11::class_<SpikingRectifiedLinear>(m, "SpikingRectifiedLinearImpl")
    //         .def(pybind11::init<>())
    //         .def("rates", &SpikingRectifiedLinear::rates,
    //             pybind11::arg("x"), pybind11::arg("gain"), pybind11::arg("bias"))
    //         .def("step", &SpikingRectifiedLinear::step,
    //             pybind11::arg("dt"), pybind11::arg("J"), pybind11::arg("output"), pybind11::arg("voltage"));
}
