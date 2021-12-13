#include "utils.h"

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& obj) {
    os << "[";
    std::for_each(obj.begin(), obj.end() - 1, [&os](const T& elem) {
        os << elem << ", ";
    });
    os << obj.back();
    os << "]";
    return os;
}
template ostream& operator<<(ostream& os, const std::vector<u32>& obj);
template ostream& operator<<(ostream& os, const std::vector<u64>& obj);
template ostream& operator<<(ostream& os, const std::vector<std::vector<u64>>& obj);

template<typename T>
ostream& operator<<(ostream& os, const unordered_set<T>& obj) {
    os << "{";
    for (auto itr = obj.begin(); itr != obj.end(); ++itr) {
        cout << *itr << ", ";
    }
    os << "}";
    return os;
}
template ostream& operator<<(ostream& os, const unordered_set<u32>& obj);

template<typename K, typename V>
ostream& operator<<(ostream& os, const std::map<K, V>& obj) {
    os << "{";
    for (auto itr = obj.begin(); itr != obj.end(); ++itr) {
        cout << "" << itr->first << ": " << itr->second << ", ";
    }
    os << "}";
    return os;
}
template ostream& operator<<(ostream& os, const std::map<u32, u32>& obj);
template ostream& operator<<(ostream& os, const std::map<u64, u64>& obj);
template ostream& operator<<(ostream& os, const std::map<u64, std::vector<std::vector<u64>>>& obj);
template ostream& operator<<(ostream& os, const std::map<u64, std::map<u64, u64>>& obj);
