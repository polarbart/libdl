
#ifndef LIBDL_UTILS_H
#define LIBDL_UTILS_H


#include <vector>
#include <optional>

class Utils {
public:
    template <typename T>
    static std::vector<T> removeOption(std::initializer_list<std::optional<T>> l) {
        std::vector<T> t;
        for (auto o : l)
            if (o.has_value())
                t.push_back(o.value());
        return t;
    }
};


#endif //LIBDL_UTILS_H
