#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

#include "../shared_arena.h"
#include "../shared_player.h"
#include "../shared_path_cache.h"
#include "../shared_point.h"

PYBIND11_MODULE(sharedcpp, share) {
    auto simulation = pybind11::module_::import("simulationcpp");

    auto shared_point = pybind11::class_<shared::SharedPoint<>>(share, "SharedPoint")
    .def_readonly_static("size", &shared::SHARED_POINT_SIZE)
    .def(pybind11::init([](const pybind11::buffer &buf) -> shared::SharedPoint<> {
        const auto buf_info = buf.request();
        assert(buf_info.size >= shared::SHARED_POINT_SIZE);
        assert(buf_info.ptr != nullptr);
        return {buf_info.ptr};
    }), pybind11::arg("buf"))
    .def_static("create", [](const pybind11::buffer &b) -> shared::SharedPoint<> {
        const auto buf_info = b.request();
        assert(buf_info.size >= shared::SHARED_POINT_SIZE);
        assert(buf_info.ptr != nullptr);
        return {buf_info.ptr, -1, -1};
    }, pybind11::arg("buf"),
    pybind11::doc("Creates a SharedPoint from the given buffer and default initializes it"))
    .def(pybind11::init<void *, simulation::PointStorageType, simulation::PointStorageType>(),
        pybind11::arg("buf"), pybind11::arg("x"), pybind11::arg("y"))
    .def(pybind11::init<void *, simulation::Point<>>(),
        pybind11::arg("buf"), pybind11::arg("p"))
    .def("magnitude", [](const shared::SharedPoint<> &t)
        { return t.get().magnitude<>(); })
    .def("direction", [](const shared::SharedPoint<> &t)
        { return t.get().direction(); })
    .def("distance", [](const shared::SharedPoint<> &t, const simulation::Point<> &p)
        { return t.get().distance(p); },
        pybind11::arg("other"))
    .def("distance", [](const shared::SharedPoint<> &t, const shared::SharedPoint<> &p)
        { return t.get().distance(p.get()); },
        pybind11::arg("other"))
    .def("copy", [](const shared::SharedPoint<> &p, const pybind11::buffer &buf) {
        const auto buf_info = buf.request();
        assert(buf_info.size >= shared::SHARED_POINT_SIZE);
        assert(buf_info.ptr != nullptr);
        return p.copy(buf_info.ptr);
    }, pybind11::arg("buf"))
    .def("clone", &shared::SharedPoint<>::clone)
    .def(pybind11::self + pybind11::self,
        pybind11::arg("other"))
    .def(pybind11::self + simulation::Point<>(int32_t(), int32_t()),
        pybind11::arg("other"))
    .def(pybind11::self + simulation::Direction(),
        pybind11::arg("other"))
    .def(pybind11::self += pybind11::self,
        pybind11::arg("other"))
    .def(pybind11::self += simulation::Point<>(int32_t(), int32_t()),
        pybind11::arg("other"))
    .def(pybind11::self += simulation::Direction(),
        pybind11::arg("other"))
    .def(pybind11::self - pybind11::self,
        pybind11::arg("other"))
    .def(pybind11::self - simulation::Point<>(int32_t(), int32_t()),
        pybind11::arg("other"))
    .def(pybind11::self -= pybind11::self,
        pybind11::arg("other"))
    .def(pybind11::self -= simulation::Point<>(int32_t(), int32_t()),
        pybind11::arg("other"))
    .def("__hash__", [](const shared::SharedPoint<> &t)
        { return std::hash<simulation::Point<>>()(t.get()); })
    .def(pybind11::self < pybind11::self,
        pybind11::arg("other"))
    .def(pybind11::self < simulation::Point<>(int32_t(), int32_t()),
        pybind11::arg("other"))
    .def(pybind11::self > pybind11::self,
        pybind11::arg("other"))
    .def(pybind11::self > simulation::Point<>(int32_t(), int32_t()),
        pybind11::arg("other"))
    .def(pybind11::self == pybind11::self,
        pybind11::arg("other"))
    .def(pybind11::self == simulation::Point<>(int32_t(), int32_t()),
        pybind11::arg("other"))
    .def(pybind11::self != pybind11::self,
        pybind11::arg("other"))
    .def(pybind11::self != simulation::Point<>(int32_t(), int32_t()),
        pybind11::arg("other"))
    .def("__str__", &shared::SharedPoint<>::operator std::string)
    .def("__repr__", &shared::SharedPoint<>::operator std::string)
#ifdef SIMULATION_JSON
    .def("__json__", &simulation::Point<>::operator nlohmann::json)
#endif
    .def_property("x", &shared::SharedPoint<>::getX, &shared::SharedPoint<>::setX)
    .def_property("y", &shared::SharedPoint<>::getY, &shared::SharedPoint<>::setY);

    auto shared_path_cache = pybind11::class_<shared::SharedPathCache<>>(share, "SharedPathCache")
    .def(pybind11::init([](const pybind11::buffer &buf) -> shared::SharedPathCache<> {
        const auto buf_info = buf.request();
        assert(buf_info.ptr != nullptr);
        return {static_cast<simulation::Point<> *>(buf_info.ptr)};
    }), pybind11::arg("buf"))
    .def(pybind11::init([](const pybind11::buffer &b, const simulation::PathCache &c) -> shared::SharedPathCache<> {
        const auto buf_info = b.request();
        assert(buf_info.size >= shared::SHARED_POINT_SIZE * c.count());
        assert(buf_info.ptr != nullptr);
        return {static_cast<simulation::Point<> *>(buf_info.ptr), c};
    }), pybind11::arg("buf"), pybind11::arg("cache"))
    .def("count", &shared::SharedPathCache<>::count)
    .def("__contains__", &shared::SharedPathCache<>::contains)
    .def("__getitem__", &shared::SharedPathCache<>::operator[])
    .def("__len__", &shared::SharedPathCache<>::size);

    auto shared_player = pybind11::class_<shared::SharedPlayer<>>(share, "SharedPlayer")
    .def_readonly_static("size", &shared::SHARED_PLAYER_SIZE)
    .def(pybind11::init([](const pybind11::buffer &buf) -> shared::SharedPlayer<> {
        const auto buf_info = buf.request();
        assert(buf_info.size >= shared::SHARED_PLAYER_SIZE);
        assert(buf_info.ptr != nullptr);
        return {buf_info.ptr};
    }))
    .def("copy", [](const pybind11::buffer &buf) -> shared::SharedPlayer<> {
        const auto buf_info = buf.request();
        assert(buf_info.size >= shared::SHARED_PLAYER_SIZE);
        assert(buf_info.ptr != nullptr);
        return {buf_info.ptr};
    })
    .def("clone", &shared::SharedPlayer<>::clone)
    .def("move", [](shared::SharedPlayer<> &t, const simulation::Direction &d)
        { return t.getPlayer().move(d); })
    .def("collect_goal", [](shared::SharedPlayer<> &t) { return t.getPlayer().collectGoal(); })
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
    .def("__str__", &shared::SharedPlayer<>::operator std::string)
    .def("__repr__", &shared::SharedPlayer<>::operator std::string)
    .def("__hash__", &std::hash<shared::SharedPlayer<>>::operator())
    .def_property("x", &shared::SharedPlayer<>::getX, &shared::SharedPlayer<>::setX)
    .def_property("y", &shared::SharedPlayer<>::getY, &shared::SharedPlayer<>::setY)
    .def_property("score", &shared::SharedPlayer<>::getScore, &shared::SharedPlayer<>::setScore)
    .def_property("positions", &shared::SharedPlayer<>::getPositions, &shared::SharedPlayer<>::setPositions);

    auto shared_arena = pybind11::class_<shared::SharedArena>(share, "SharedArena")
    .def_readonly_static("size", &shared::SHARED_ARENA_SIZE)
    .def(pybind11::init([](const pybind11::buffer &buf) -> shared::SharedArena {
        const auto buf_info = buf.request();
        assert(buf_info.size >= shared::SHARED_ARENA_SIZE);
        assert(buf_info.ptr != nullptr);
        return {buf_info.ptr};
    }))
    .def(pybind11::init([](const pybind11::buffer &buf, const simulation::Arena<> &a) -> shared::SharedArena {
        const auto buf_info = buf.request();
        assert(buf_info.size >= shared::SHARED_ARENA_SIZE);
        assert(buf_info.ptr != nullptr);
        return {buf_info.ptr, a};
    }))
    .def("on_goal", [](shared::SharedArena &t) { return t.get().onGoal(); })
    .def("best_direction", [](shared::SharedArena &t, const shared::SharedPoint<> &start, const shared::SharedPoint<> &end)
        { return t.get().bestDirection(start.get(), end.get()); })
    .def("best_direction", [](shared::SharedArena &t, const simulation::Point<> &start, const simulation::Point<> &end)
        { return t.get().bestDirection(start, end); })
    .def("set_goal", [](shared::SharedArena &t) { return t.get().setGoal(); })
    .def("detection", [](shared::SharedArena &t) { return t.get().detection(); })
    .def("absolute_distance", [](shared::SharedArena &t){ return t.get().absoluteDistance(); })
    .def("distance", [](shared::SharedArena &t) { return t.get().distance(); })
    .def("distance", [](shared::SharedArena &t, const simulation::Point<> &a)
        { return t.get().distance(a); })
    .def("distance", [](shared::SharedArena &t, const simulation::Point<> &a, const simulation::Point<> &b)
        { return t.get().distance(a, b); })
    .def("distance", [](shared::SharedArena &t, const shared::SharedPoint<> &a)
        { return t.get().distance(a.get()); })
    .def("distance", [](shared::SharedArena &t, const shared::SharedPoint<> &a, const shared::SharedPoint<> &b)
        { return t.get().distance(a.get(), b.get()); })
    .def("__str__", &shared::SharedArena::operator std::string)
    .def("__repr__", &shared::SharedArena::operator std::string)
    .def_property_readonly("n", [] {return 23;})
    .def_property_readonly("m", [] {return 23;})
    .def_property_readonly("goal",
        [](shared::SharedArena &t) { return t.get().getGoal(); })
    .def_property_readonly("grid",
        [](shared::SharedArena &t) { return t.get().getGrid(); })
    .def_property("player",
        [](shared::SharedArena &t) { return t.get().getPlayer(); },
        [](shared::SharedArena &t, const simulation::Player<> &p) { t.get().setPlayer(p); });
}