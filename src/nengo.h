#ifndef NENGOCPP_LIBRARY_H
#define NENGOCPP_LIBRARY_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <eigen3/Eigen/Core>

namespace nengo::neuron_types {

    using ValueType = double;
    using IndexType = int32_t;

    // Matrix with numpy storage order and ValueType values with a given size
    template<IndexType rows, IndexType cols>
    using Matrix = Eigen::Matrix<ValueType, rows, cols, Eigen::RowMajor>;
    template<IndexType rows, IndexType cols>
    using MatrixRef = Eigen::Ref<Matrix<rows, cols>>;
    template<IndexType rows, IndexType cols>
    using ConstMatrixRef = Eigen::Ref<const Matrix<rows, cols>>;

    using GenericMatrix = Matrix<Eigen::Dynamic, Eigen::Dynamic>;
    using GenericMatrixRef = MatrixRef<Eigen::Dynamic, Eigen::Dynamic>;
    using ConstGenericMatrixRef = ConstMatrixRef<Eigen::Dynamic, Eigen::Dynamic>;

    template <IndexType size>
    using RowVector = Eigen::Matrix<ValueType, 1, size, Eigen::RowMajor>;
    template <IndexType size>
    using RowVectorRef = Eigen::Ref<RowVector<size>>;
    template <IndexType size>
    using ConstRowVectorRef = Eigen::Ref<const RowVector<size>>;

    template <IndexType size>
    using ColumnVector = Eigen::Vector<ValueType, size>;
    template <IndexType size>
    using ColumnVectorRef = Eigen::Ref<ColumnVector<size>>;
    template <IndexType size>
    using ConstColumnVectorRef = Eigen::Ref<const ColumnVector<size>>;

    using GenericRowVector = RowVector<Eigen::Dynamic>;
    using GenericRowVectorRef = RowVectorRef<Eigen::Dynamic>;
    using ConstGenericRowVectorRef = ConstRowVectorRef<Eigen::Dynamic>;

    using GenericColumnVector = ColumnVector<Eigen::Dynamic>;
    using GenericColumnVectorRef = ColumnVectorRef<Eigen::Dynamic>;
    using ConstGenericColumnVectorRef = ConstColumnVectorRef<Eigen::Dynamic>;


    struct NeuronType {
        // Compute current injected in each neuron given input, gain and bias.
        // This is the case when the input is given as a 2D Matrix
        // In this case gain and bias are broadcast to the input
        template<auto n_samples, auto n_neurons>
        Matrix<n_samples, n_neurons> current(__attribute__((unused)) const pybind11::object &self,
            const Matrix<n_samples, n_neurons> &x,
            const ColumnVector<n_neurons> &gain,
            const ColumnVector<n_neurons> &bias) {
            // Validation check to ensure the dimensions are correct when the templated size is dynamic
            if (n_neurons == Eigen::Dynamic and x.cols() != gain.size()) {
                auto x_rows = x.rows();
                auto x_cols = x.cols();
                auto g_sze = gain.size();
                std::stringstream ss;
                ss << "Expected shape (" << x_rows << ", " << g_sze << "); got (" << x_rows << ", " << x_cols << ").";
                throw std::invalid_argument(ss.str());
            }
            // Broadcast the coefficient wise product and sum to the gain and bias
            // (Reduced to one line to allow the compiler to optimize the operations)
            return (x.array().rowwise() * gain.transpose().array()).array().rowwise() + bias.transpose().array();
        }
        // Same function as above but only when the input is just a vector
        // The input will be cast to a 2D matrix before being operated on
        template<auto n_neurons>
        Matrix<n_neurons, 1> current(const pybind11::object &self,
            const ColumnVector<n_neurons> &x,
            const ColumnVectorRef<n_neurons> &gain,
            const ColumnVectorRef<n_neurons> &bias) {
            // add the extra dimension for x to satisfy the above function signature
            return current<n_neurons, 1>(self, x, gain, bias);
        }

        template<auto n_neurons>
        std::pair<RowVector<n_neurons>, RowVector<n_neurons>> gain_bias(__attribute__((unused)) const pybind11::object &self,
            const ColumnVector<n_neurons> &max_rates,
            const ColumnVector<n_neurons> &intercepts) {

        }
    };

    constexpr uint32_t N_NEURONS = 1000;
/*
    struct RectifiedLinear {
        RectifiedLinear() = default;
        static pybind11::tuple gain_bias(__attribute__((unused)) const pybind11::object &self, const NeuronType::GenericNeuralVector &max_rates, const NeuronType::GenericNeuralVector &intercepts);
        static pybind11::tuple max_rates_intercepts(__attribute__((unused)) const pybind11::object &self, const NeuronType::GenericNeuralVector &gain, const NeuronType::GenericNeuralVector &bias);
        static void step(const pybind11::object &self, float dt, const NeuronType::GenericNeuralMatrix &J, NeuronType::GenericNeuralMatrix &output);
    };

    struct SpikingRectifiedLinear {
        SpikingRectifiedLinear() = default;
        static Eigen::MatrixXd rates(pybind11::object &self, const NeuronType::GenericNeuralMatrix &x, const NeuronType::GenericNeuralMatrix &gain, const NeuronType::GenericNeuralMatrix &bias);
        static void step(const pybind11::object &self, float dt, const NeuronType::GenericNeuralMatrix &J, NeuronType::GenericNeuralMatrix &output, NeuronType::GenericNeuralMatrix &voltage);
    };
*/
}

#endif //NENGOCPP_LIBRARY_H