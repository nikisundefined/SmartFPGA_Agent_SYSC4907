#ifndef NENGOCPP_LIBRARY_H
#define NENGOCPP_LIBRARY_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

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
    using RowVector = Eigen::Matrix<ValueType, size, 1, Eigen::RowMajor>;
    template <IndexType size>
    using RowVectorRef = Eigen::Ref<RowVector<size>>;
    template <IndexType size>
    using ConstRowVectorRef = Eigen::Ref<const RowVector<size>>;

    template <IndexType size>
    using ColumnVector = Eigen::Matrix<ValueType, 1, size, Eigen::ColMajor>;
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
        // The format of a matrix representing the neurons in an ensemble with a set number
        template <uint32_t neurons>
        using NeuralMatrix = Eigen::Ref<Eigen::Matrix<double, neurons, 1, Eigen::RowMajor>>;

        // Represents a dynamic implementation of a Neural Matrix
        using GenericNeuralMatrix = Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
        using ConstGenericNeuralMatrix = Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

        using GenericNeuralVector = Eigen::Ref<Eigen::Vector<double, Eigen::Dynamic>>;
        using ConstGenericNeuralVector = Eigen::Ref<const Eigen::Vector<double, Eigen::Dynamic>>;

        // The function signature for the gain_bias calculation optimized for the given number of neurons
        template <uint32_t n_neurons>
        using gain_bias = std::function<pybind11::tuple(const pybind11::object &, const NeuralMatrix<n_neurons> &, const NeuralMatrix<n_neurons> &)>;

        // The function signature for the max_rates_intercepts calculation optimized for the given number of neurons
        template <uint32_t n_neurons>
        using max_rates_intercepts = std::function<pybind11::tuple(const pybind11::object &, const NeuralMatrix<n_neurons> &, const NeuralMatrix<n_neurons> &)>;

        template <uint32_t n_neurons>
        using step = std::function<void(pybind11::object &, float, const NeuralMatrix<n_neurons> &, NeuralMatrix<n_neurons> &, pybind11::kwargs &)>;

        template<auto n_samples, auto n_neurons>
        static Matrix<n_samples, n_neurons> current(__attribute__((unused)) const pybind11::object &self, const MatrixRef<n_samples, n_neurons> &x, const RowVectorRef<n_neurons> &gain, const RowVectorRef<n_neurons> &bias);
    };

    constexpr uint32_t N_NEURONS = 1000;

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

}

#endif //NENGOCPP_LIBRARY_H