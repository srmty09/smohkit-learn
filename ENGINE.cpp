#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <numeric>

namespace py = pybind11;
typedef long long ll;
using namespace std;

class dumpy {
private:
    vector<vector<double>*> vec_ptrs;

public:
    vector<double>* array(vector<double>& vec) {
        vector<double>* data = new vector<double>(vec);
        vec_ptrs.push_back(data);
        return data;
    }

    void print(vector<double>* ptr) {
        if (!ptr) throw invalid_argument("Null pointer in print");
        vector<double>& data = *ptr;
        cout << "[";
        for (size_t i = 0; i < data.size(); ++i) {
            cout << data[i];
            if (i+1 != data.size()) cout << ", ";
        }
        cout << "] (type: dumpy array)" << endl;
    }

    vector<double>* zeros(ll n) {
        vector<double>* vec = new vector<double>(n, 0.0);
        vec_ptrs.push_back(vec);
        return vec;
    }

    vector<double>* ones(ll n) {
        vector<double>* vec = new vector<double>(n, 1.0);
        vec_ptrs.push_back(vec);
        return vec;
    }

    vector<double>* arange(double start, double end, double step=1.0) {
        if (step == 0) throw invalid_argument("Step cannot be zero");
        vector<double>* vec = new vector<double>();
        if (step > 0) {
            for (double val = start; val < end; val += step)
                vec->push_back(val);
        } else {
            for (double val = start; val > end; val += step)
                vec->push_back(val);
        }
        vec_ptrs.push_back(vec);
        return vec;
    }

    vector<double>* linspace(double start, double end, ll nums) {
        if (nums <= 1) throw invalid_argument("nums must be > 1");
        vector<double>* vec = new vector<double>();
        double step = (end - start) / (nums - 1);
        for (ll i = 0; i < nums; i++)
            vec->push_back(start + i * step);
        vec_ptrs.push_back(vec);
        return vec;
    }

    vector<double>* add(vector<double>* a, vector<double>* b) {
        check_null(a, b);
        vector<double>&va=*a;
        vector<double>&vb=*b;
        size_t n = min(va.size(), vb.size());
        vector<double>* res = new vector<double>();
        for (size_t i = 0; i < n; i++)
            res->push_back(va[i] + vb[i]);
        vec_ptrs.push_back(res);
        return res;
    }

    vector<double>* sub(vector<double>* a, vector<double>* b) {
        check_null(a, b);
        vector<double>&va=*a;
        vector<double>&vb=*b;
        size_t n = min(va.size(), vb.size());
        vector<double>* res = new vector<double>();
        for (size_t i = 0; i < n; i++)
            res->push_back(va[i] - vb[i]);
        vec_ptrs.push_back(res);
        return res;
    }

    vector<double>* mul(vector<double>* a, vector<double>* b) {
        check_null(a, b);
        vector<double>&va=*a;
        vector<double>&vb=*b;
        size_t n = min(va.size(), vb.size());
        vector<double>* res = new vector<double>();
        for (size_t i = 0; i < n; i++)
            res->push_back(va[i] * vb[i]);
        vec_ptrs.push_back(res);
        return res;
    }

    vector<double>* div(vector<double>* a, vector<double>* b) {
        check_null(a, b);
        vector<double>&va=*a;
        vector<double>&vb=*b;
        size_t n = min(va.size(), vb.size());
        vector<double>* res = new vector<double>();
        for (size_t i = 0; i < n; i++) {
            if (vb[i] == 0) throw invalid_argument("Division by zero");
            res->push_back(va[i] / vb[i]);
        }
        vec_ptrs.push_back(res);
        return res;
    }

    double dot(vector<double>* a, vector<double>* b) {
        check_null(a, b);
        vector<double>& va = *a;
        vector<double>& vb = *b;
        size_t n = min(va.size(), vb.size());
        double res = 0;
        for (size_t i = 0; i < n; i++)
            res += va[i] * vb[i];
        return res;
    }

    double mean(vector<double>* a) {
        check_null(a);
        if (a->empty()) throw invalid_argument("Empty vector in mean");
        double sum = accumulate(a->begin(), a->end(), 0.0);
        return sum / a->size();
    }

    double median(vector<double>* a) {
        check_null(a);
        if (a->empty()) throw invalid_argument("Empty vector in median");
        vector<double> tmp = *a;
        std::sort(tmp.begin(), tmp.end());
        size_t n = tmp.size();
        if (n % 2 == 0) {
            return (tmp[n/2 - 1] + tmp[n/2]) / 2.0;
        } else {
            return tmp[n/2];
        }
    }

    double std(vector<double>* a) {
        check_null(a);
        if (a->empty()) throw invalid_argument("Empty vector in std");
        double m = mean(a);
        double accum = 0;
        for (auto val : *a)
            accum += (val - m) * (val - m);
        return std::sqrt(accum / a->size());
    }

    double var(vector<double>* a) {
        check_null(a);
        double s = std(a);
        return s * s;
    }

    vector<double>* sort(vector<double>* a) {
        check_null(a);
        vector<double>* res = new vector<double>(*a);
        std::sort(res->begin(), res->end());
        vec_ptrs.push_back(res);
        return res;
    }

    vector<double>* unique(vector<double>* a) {
        check_null(a);
        vector<double> tmp = *a;
        std::sort(tmp.begin(), tmp.end());
        tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
        vector<double>* res = new vector<double>(tmp);
        vec_ptrs.push_back(res);
        return res;
    }

    double max(vector<double>* a) {
        check_null(a);
        if (a->empty()) throw invalid_argument("Empty vector in max");
        return *std::max_element(a->begin(), a->end());
    }

    double min_element(vector<double>* a) {
        check_null(a);
        if (a->empty()) throw invalid_argument("Empty vector in min");
        return *std::min_element(a->begin(), a->end());
    }

    double sum(vector<double>* a) {
        check_null(a);
        return accumulate(a->begin(), a->end(), 0.0);
    }

    vector<double>* abs(vector<double>* a) {
        check_null(a);
        vector<double>* res = new vector<double>();
        for (auto v : *a) res->push_back(std::abs(v));
        vec_ptrs.push_back(res);
        return res;
    }

    vector<double>* pow(vector<double>* a, double exponent) {
        check_null(a);
        vector<double>* res = new vector<double>();
        for (auto v : *a) res->push_back(std::pow(v, exponent));
        vec_ptrs.push_back(res);
        return res;
    }

    vector<double>* sqrt(vector<double>* a) {
        check_null(a);
        vector<double>* res = new vector<double>();
        for (auto v : *a) {
            if (v < 0) throw invalid_argument("Negative value for sqrt");
            res->push_back(std::sqrt(v));
        }
        vec_ptrs.push_back(res);
        return res;
    }

    vector<double>* floor(vector<double>* a) {
        check_null(a);
        vector<double>* res = new vector<double>();
        for (auto v : *a) res->push_back(std::floor(v));
        vec_ptrs.push_back(res);
        return res;
    }

    vector<double>* ceil(vector<double>* a) {
        check_null(a);
        vector<double>* res = new vector<double>();
        for (auto v : *a) res->push_back(std::ceil(v));
        vec_ptrs.push_back(res);
        return res;
    }

    vector<double>* round(vector<double>* a) {
        check_null(a);
        vector<double>* res = new vector<double>();
        for (auto v : *a) res->push_back(std::round(v));
        vec_ptrs.push_back(res);
        return res;
    }

    vector<double>* clip(vector<double>* a, double low, double high) {
        check_null(a);
        if (low > high) throw invalid_argument("low must be <= high in clip");
        vector<double>* res = new vector<double>();
        for (auto v : *a) {
            if (v < low) res->push_back(low);
            else if (v > high) res->push_back(high);
            else res->push_back(v);
        }
        vec_ptrs.push_back(res);
        return res;
    }

    vector<double>* logical_and(vector<double>* a, vector<double>* b) {
        check_null(a, b);
        size_t n = min(a->size(), b->size());
        vector<double>* res = new vector<double>();
        for (size_t i = 0; i < n; i++)
            res->push_back((*a)[i] != 0 && (*b)[i] != 0 ? 1.0 : 0.0);
        vec_ptrs.push_back(res);
        return res;
    }

    vector<double>* logical_or(vector<double>* a, vector<double>* b) {
        check_null(a, b);
        size_t n = min(a->size(), b->size());
        vector<double>* res = new vector<double>();
        for (size_t i = 0; i < n; i++)
            res->push_back((*a)[i] != 0 || (*b)[i] != 0 ? 1.0 : 0.0);
        vec_ptrs.push_back(res);
        return res;
    }

    vector<double>* logical_not(vector<double>* a) {
        check_null(a);
        vector<double>* res = new vector<double>();
        for (auto v : *a) res->push_back(v == 0 ? 1.0 : 0.0);
        vec_ptrs.push_back(res);
        return res;
    }

    vector<double>* exp(vector<double>* a) {
        check_null(a);
        vector<double>* res = new vector<double>();
        for (auto v : *a) res->push_back(std::exp(v));
        vec_ptrs.push_back(res);
        return res;
    }

    vector<double>* log(vector<double>* a) {
        check_null(a);
        vector<double>* res = new vector<double>();
        for (auto v : *a) {
            if (v <= 0) throw invalid_argument("Log domain error: non-positive value");
            res->push_back(std::log(v));
        }
        vec_ptrs.push_back(res);
        return res;
    }

private:
    void check_null(vector<double>* a) {
        if (!a) throw invalid_argument("Null pointer passed");
    }
    void check_null(vector<double>* a, vector<double>* b) {
        if (!a || !b) throw invalid_argument("Null pointer passed");
    }

public:
    ~dumpy() {
        for (auto p : vec_ptrs) delete p;
    }
};

PYBIND11_MODULE(ENGINE, m) {
    py::class_<dumpy>(m, "dumpy")
        .def(py::init<>())
        .def("array", &dumpy::array, py::return_value_policy::reference)
        .def("print", &dumpy::print)
        .def("zeros", &dumpy::zeros, py::return_value_policy::reference)
        .def("ones", &dumpy::ones, py::return_value_policy::reference)
        .def("arange", &dumpy::arange, py::return_value_policy::reference)
        .def("linspace", &dumpy::linspace, py::return_value_policy::reference)
        .def("sum", &dumpy::sum)
        .def("mean", &dumpy::mean)
        .def("median", &dumpy::median)
        .def("std", &dumpy::std)
        .def("var", &dumpy::var)
        .def("max", &dumpy::max)
        .def("min", &dumpy::min_element)
        .def("sort", &dumpy::sort, py::return_value_policy::reference)
        .def("unique", &dumpy::unique, py::return_value_policy::reference)
        .def("add", &dumpy::add, py::return_value_policy::reference)
        .def("sub", &dumpy::sub, py::return_value_policy::reference)
        .def("mul", &dumpy::mul, py::return_value_policy::reference)
        .def("div", &dumpy::div, py::return_value_policy::reference)
        .def("dot", &dumpy::dot)
        .def("abs", &dumpy::abs, py::return_value_policy::reference)
        .def("pow", &dumpy::pow, py::return_value_policy::reference)
        .def("sqrt", &dumpy::sqrt, py::return_value_policy::reference)
        .def("floor", &dumpy::floor, py::return_value_policy::reference)
        .def("ceil", &dumpy::ceil, py::return_value_policy::reference)
        .def("round", &dumpy::round, py::return_value_policy::reference)
        .def("clip", &dumpy::clip, py::return_value_policy::reference)
        .def("logical_and", &dumpy::logical_and, py::return_value_policy::reference)
        .def("logical_or", &dumpy::logical_or, py::return_value_policy::reference)
        .def("logical_not", &dumpy::logical_not, py::return_value_policy::reference)
        .def("exp", &dumpy::exp, py::return_value_policy::reference)
        .def("log", &dumpy::log, py::return_value_policy::reference);
}
