/** \file params_py.cpp
 */

#include "params_py.h"

#include <pybind11/eval.h>
#include "pybind11/stl.h"

namespace trimem
{

std::string err = "Initializing ContinuationTuple from list failed; "
"Must be either [start, stop, delta, lambda] or"
"[string-expr, N, delta, lambda, <optional-label>";

std::vector<real> eval_expr(
    std::string& expr,
    int N,
    std::optional<const py::str> label
)
{
    py::dict scope = py::module_::import("math").attr("__dict__");
    scope["linspace"] = py::module_::import("numpy").attr("linspace");
    std::string eval = "[eval('" + expr + "') " + \
                       "for x in linspace(0,1," + \
                       std::to_string(N) + ")]";
    py::list res = py::eval(eval, scope);

    if (label.has_value())
    {
        py::object conf = py::module_::import("trimem.mc.config");
        py::object plot = conf.attr("termplot");
        plot(res, label);
    }

    return res.cast<std::vector<real>>();
}

ContinuationTuple make_continuation_from_list(const py::list& args)
{
    int l = py::len(args);
    if (l == 1)
    {
        real val = args[0].cast<real>();
        return ContinuationTuple(args[0].cast<real>());
    }
    else if (l >= 4)
    {
        try
        {
            real start = args[0].cast<real>();
            return ContinuationTuple(
                start,
                args[1].cast<real>(),
                args[2].cast<real>(),
                args[3].cast<real>()
            );
        }
        catch (...)
        {
            try
            {
                auto expr = args[0].cast<std::string>();
                int  N    = args[1].cast<int>();

                std::optional<py::str> label = std::nullopt;
                if (l == 5)
                    label = args[4].cast<py::str>();

                auto data = eval_expr(expr, N, label);
                return ContinuationTuple(
                    data,
                    expr,
                    args[2].cast<real>(),
                    args[3].cast<real>(),
                    label
                );
            }
            catch (...)
            {
                throw std::runtime_error(err);
            }
        }
    }
    else
        throw std::runtime_error(err);
}

}
