#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>

#include "../direction.h"
#include "../point.h"
#include "../player.h"
#include "../path_pair.h"
#include "../path_cache.h"
#include "../arena.h"
#include "../pathfinding.h"

PYBIND11_MODULE(simulationcpp, sim) {
    auto direction = pybind11::enum_<simulation::Direction>(sim, "Direction", pybind11::module_local())
    .value("UP", simulation::Direction::Up)
    .value("DOWN", simulation::Direction::Down)
    .value("LEFT", simulation::Direction::Left)
    .value("RIGHT", simulation::Direction::Right)
    .value("NONE", simulation::Direction::None)
    .export_values();

    auto point = pybind11::class_<simulation::Point<>>(sim, "Point")
    .def(pybind11::init<simulation::PointStorageType, simulation::PointStorageType>(),
        pybind11::arg("x"), pybind11::arg("y"))
    .def(pybind11::init<Eigen::Ref<const Eigen::Vector<simulation::PointStorageType, 4>>>(),
        pybind11::arg("vec"))
    .def("magnitude", &simulation::Point<>::magnitude<>)
    .def("direction", &simulation::Point<>::direction)
    .def("distance", &simulation::Point<>::distance,
        pybind11::arg("other"))
    .def("copy", &simulation::Point<>::copy)
    .def("clone", &simulation::Point<>::clone)
    .def(pybind11::self + pybind11::self,
        pybind11::arg("other"))
    .def(pybind11::self + simulation::Direction(),
        pybind11::arg("other"))
    .def(pybind11::self += pybind11::self,
        pybind11::arg("other"))
    .def(pybind11::self += simulation::Direction(),
        pybind11::arg("other"))
    .def(pybind11::self - pybind11::self,
        pybind11::arg("other"))
    .def(pybind11::self -= pybind11::self,
        pybind11::arg("other"))
    .def("__hash__", &std::hash<simulation::Point<>>::operator())
    .def("__lt__", &simulation::Point<>::operator<,
        pybind11::arg("other"))
    .def("__gt__", &simulation::Point<>::operator>,
        pybind11::arg("other"))
    .def("__eq__", &simulation::Point<>::operator==,
        pybind11::arg("other"))
    .def("__str__", &simulation::Point<>::operator std::string)
    .def("__repr__", &simulation::Point<>::operator std::string)
#ifdef SIMULATION_JSON
    .def("__json__", &simulation::Point<>::operator nlohmann::json)
#endif
    .def_property("x", &simulation::Point<>::getX, &simulation::Point<>::setX)
    .def_property("y", &simulation::Point<>::getY, &simulation::Point<>::setY);

    auto player = pybind11::class_<simulation::Player<>>(sim, "Player")
    .def(pybind11::init<const simulation::Point<> &>(),
        pybind11::arg("point"))
    .def(pybind11::init<const simulation::Point<> &, uint32_t>(),
        pybind11::arg("point"), pybind11::arg("score"))
    .def(pybind11::init<const simulation::Point<> &, uint32_t, const std::list<simulation::Point<>> &>(),
        pybind11::arg("point"), pybind11::arg("score"), pybind11::arg("positions"))
    .def("copy", &simulation::Player<>::copy)
    .def("clone", &simulation::Player<>::clone)
    .def("move", &simulation::Player<>::move,
        pybind11::arg("direction"))
    .def("collect_goal", &simulation::Player<>::collectGoal)
    .def(pybind11::self + simulation::Point<>(uint32_t(), uint32_t()),
        pybind11::arg("other"))
    .def(pybind11::self + simulation::Direction(),
        pybind11::arg("other"))
    .def(pybind11::self - simulation::Point<>(uint32_t(), uint32_t()),
        pybind11::arg("other"))
    .def(pybind11::self += simulation::Point<>(uint32_t(), uint32_t()),
        pybind11::arg("other"))
    .def(pybind11::self += simulation::Direction(),
        pybind11::arg("other"))
    .def("__str__", &simulation::Player<>::operator std::string)
    .def("__repr__", &simulation::Player<>::operator std::string)
    .def("__hash__", &std::hash<simulation::Player<>>::operator())
#ifdef SIMULATION_JSON
    .def("__json__", &simulation::Player<>::operator nlohmann::json)
#endif
    .def_property("x", &simulation::Player<>::getX, &simulation::Player<>::setX)
    .def_property("y", &simulation::Player<>::getY, &simulation::Player<>::setY)
    .def_property("score", &simulation::Player<>::getScore, &simulation::Player<>::setScore)
    .def_property("positions", &simulation::Player<>::getPositions, &simulation::Player<>::setPositions);

    auto path_pair = pybind11::class_<simulation::PathPair>(sim, "PathPair")
    .def(pybind11::init<const simulation::Point<> &, const simulation::Point<> &>())
    .def(pybind11::init<std::string_view>())
    .def("__eq__", &simulation::PathPair::operator==)
    .def("__str__", &simulation::PathPair::operator std::string)
    .def("__repr__", &simulation::PathPair::operator std::string)
    .def("__hash__", &std::hash<simulation::PathPair>::operator())
    .def_property_readonly("start", &simulation::PathPair::getStart)
    .def_property_readonly("end", &simulation::PathPair::getEnd);

    auto path_cache = pybind11::class_<simulation::PathCache>(sim, "PathCache")
    .def(pybind11::init<>())
#ifdef SIMULATION_JSON
    .def(pybind11::init<std::string_view>())
    .def(pybind11::init<std::filesystem::path>())
    .def("__json__", &simulation::PathCache::operator nlohmann::json)
#endif
    .def("count", &simulation::PathCache::count)
    .def("__contains__", &simulation::PathCache::contains)
    .def("__getitem__", &simulation::PathCache::operator[])
    .def("__setitem__", [](const simulation::PathCache &self, const simulation::PathPair &k, const std::list<simulation::Point<>> &v) {
        self[k] = v;
    })
    .def("__len__", &simulation::PathCache::size);

    auto arena_tile = pybind11::enum_<simulation::ArenaTile>(sim, "ArenaTile")
    .value("EMPTY", simulation::ArenaTile::Empty)
    .value("WALL", simulation::ArenaTile::Wall)
    .value("PLAYER", simulation::ArenaTile::Player)
    .value("GOAL", simulation::ArenaTile::Goal)
    .export_values();

    auto arena = pybind11::class_<simulation::Arena<>>(sim, "Arena")
    .def(pybind11::init<>())
    .def("on_goal", &simulation::Arena<>::onGoal)
    .def("best_direction", &simulation::Arena<>::bestDirection)
    .def("move", &simulation::Arena<>::move)
    .def("set_goal", &simulation::Arena<>::setGoal)
    .def("detection", &simulation::Arena<>::detection)
    .def("absolute_distance", &simulation::Arena<>::absoluteDistance)
    .def("distance", (std::list<simulation::Point<>>(simulation::Arena<>::*)(const simulation::Point<> &, const simulation::Point<> &) const)&simulation::Arena<>::distance)
    .def("__str__", &simulation::Arena<>::operator std::string)
    .def("__repr__", &simulation::Arena<>::operator std::string)
#ifdef SMART_AGENT_JSON
    .def("__json__", &simulation::Arena<>::operator nlohmann::json)
#endif
    .def_property_readonly("n", [] {return 23;})
    .def_property_readonly("m", [] {return 23;})
    .def_property_readonly("goal", &simulation::Arena<>::getGoal)
    .def_property_readonly("grid", &simulation::Arena<>::getGrid)
    .def_property("player", &simulation::Arena<>::getPlayer, &simulation::Arena<>::setPlayer);
}