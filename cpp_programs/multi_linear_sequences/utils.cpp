#include "utils.h"

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& obj) {
    os << "[";
    for (auto itr = obj.begin(); itr != obj.end(); ++itr) {
        cout << *itr << ", ";
    }
    os << "]";
    return os;
}
template ostream& operator<<(ostream& os, const std::vector<U32>& obj);
template ostream& operator<<(ostream& os, const std::vector<U64>& obj);
template ostream& operator<<(ostream& os, const std::vector<std::vector<U64>>& obj);

template<typename T>
ostream& operator<<(ostream& os, const unordered_set<T>& obj) {
    os << "{";
    for (auto itr = obj.begin(); itr != obj.end(); ++itr) {
        cout << *itr << ", ";
    }
    os << "}";
    return os;
}
template ostream& operator<<(ostream& os, const unordered_set<U32>& obj);

template<typename K, typename V>
ostream& operator<<(ostream& os, const std::map<K, V>& obj) {
    os << "{";
    for (auto itr = obj.begin(); itr != obj.end(); ++itr) {
        cout << "" << itr->first << ": " << itr->second << ", ";
    }
    os << "}";
    return os;
}
template ostream& operator<<(ostream& os, const std::map<U32, U32>& obj);
template ostream& operator<<(ostream& os, const std::map<U64, U64>& obj);
template ostream& operator<<(ostream& os, const std::map<U64, std::vector<std::vector<U64>>>& obj);
template ostream& operator<<(ostream& os, const std::map<U64, std::map<U64, U64>>& obj);
template ostream& operator<<(ostream& os, const std::map<U32, std::map<U32, U32>>& obj);
