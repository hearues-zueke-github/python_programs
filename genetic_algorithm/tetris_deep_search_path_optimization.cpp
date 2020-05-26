#include <iostream>
#include <vector>
#include <stdint.h>
#include <ostream>
#include <sstream>
#include <string>
#include <map>
#include <tuple>
#include <iterator>
#include <random>
#include <exception>

#include "../cpp_programs/primes/utils.h"

// using namespace std;
using std::cout;
using std::endl;
using std::vector;
using std::fill;
using std::ostream;
using std::stringstream;
using std::ostringstream;
using std::for_each;
using std::string;
using std::map;
using std::tuple;

template<typename Iter, typename RandomGenerator>
Iter select_randomly(Iter start, Iter end, RandomGenerator& g) {
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(g));
    return start;
}

template<typename Iter>
Iter select_randomly(Iter start, Iter end) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return select_randomly(start, end, gen);
}

class TetrisField : public std::exception {
private:
  uint8_t* field_arr_;
  uint8_t** field_;
  
  int rows_;
  int columns_;

  typedef std::tuple<string, string, int> key_name_dir_pos;

  struct Position {
    int rows_;
    int columns_;
    // int* pos_arr_;
    // int** pos_;
    vector<vector<int>> pos_;

    Position() {
    }

    void setPositonParams(const int rows, const int columns) {
    // Position (const int rows, const int columns) {
      rows_ = rows;
      columns_ = columns;
      // pos_arr_ = new int[rows * columns];
      // pos_ = new int*[rows];
      // for (int i = 0; i < rows; ++i) {
      //   pos_[i] = pos_arr_ + columns * i;
      // }
    }

    ~Position() {
      // if (pos_arr_ != nullptr) {
      //   delete[] pos_arr_;
      //   pos_arr_ = nullptr;
      // }
      // if (pos_arr_ != nullptr) {
      //   delete[] pos_;
      //   pos_ = nullptr;
      // }
    }

    // TODO: fix this const later!
    void copyPos(const vector<vector<int>>& pos) {
      pos_ = pos;

      // for (int j = 0; j < rows_; ++j) {
      //   for (int i = 0; i < columns_; ++i) {
      //     pos_arr_[j * columns_ + i] = pos_arr[j][i];
      //     // cout << "pos_arr_[i]: " << pos_arr_[i] << ", pos_arr[i]: " << pos_arr[i] << endl;
      //   }
      // }
      // std::copy(&pos_arr, &pos_arr + rows_ * columns_, &pos_arr_);
    }

    void copyPos(const Position& obj) {
      pos_ = obj.pos_;
      // std::copy(&obj.pos_arr_, &obj.pos_arr_ + rows_ * columns_, &pos_arr_);
    }

    void printPos() {
      ostringstream ss;
      cout << "pos_: " << pos_ << endl;
      for (int i = 0; i < rows_; ++i) {
        // ss << "(" << pos_arr_[i*2+0] << ", " << pos_arr_[i*2+1] << "), ";
        ss << "(" << pos_[i][0] << ", " << pos_[i][1] << "), ";
      }
      string s = ss.str();
      s.resize(s.size() - 2);
      cout << "pos_: " << s << endl;
    }
  };

  struct Piece : public Position {
    key_name_dir_pos key_;
    bool is_piece_set_;
    string name_;
    string direction_;
    int pos_place_;
    int idx_;

    Piece() {
    }
  };
  Piece current_piece_;

  struct PiecePositions : public Position {
    string name_;
    string direction_;
    int min_x_;
    int max_x_;
    int idx_;

    PiecePositions() {
    }

    PiecePositions(string name, string direction, int min_x, int max_x) :
        name_(name), direction_(direction), min_x_(min_x), max_x_(max_x) {
    }

    void setPiecePosition(const vector<vector<int>>& pos) {
      const int rows = pos.size();
      const int columns= pos[0].size();
      setPositonParams(rows, columns);
      copyPos(pos);
    }

    ~PiecePositions() {
    }

    void printProperties() {
      ostringstream ss;
      ss << "1 name_: " << name_ << endl;
      ss << "2 direction_: " << direction_ << endl;
      ss << "3 min_x_: " << min_x_ << endl;
      ss << "4 max_x_: " << max_x_ << endl;
      ss << "5 idx_: " << idx_ << endl;
      string s = ss.str();
      cout << s;
    }
  };
  PiecePositions piece_positions_[19];

  struct AbsStartPiecePositions : public Position {
    AbsStartPiecePositions() {
    }
  };
  
  vector<string> pieces_name_ = {"T", "O", "L", "S", "Z", "J", "I"};
  vector<uint8_t> pieces_idx_ = {1, 2, 3, 4, 5, 6, 7};
  map<string, uint8_t> piece_name_to_idx_;
  map<string, uint8_t> piece_name_to_index_;

  map<key_name_dir_pos, AbsStartPiecePositions> piece_name_direction_pos_to_abs_pos_;
  vector<key_name_dir_pos> keys_;
public:
  TetrisField(const int rows, const int columns) {
    current_piece_.is_piece_set_ = false;

    rows_ = rows;
    columns_ = columns;

    field_arr_ = new uint8_t[rows * columns];
    field_ = new uint8_t*[rows];

    for (int i = 0; i < rows; ++i) {
      field_[i] = &(field_arr_[i * columns]);
    }

    for (int i = 0; i < 7; ++i) {
      piece_name_to_index_[pieces_name_[i]] = i;
      piece_name_to_idx_[pieces_name_[i]] = pieces_idx_[i];
    }

    piece_positions_[ 0] = {"J", "N", 0, columns_-2};
    piece_positions_[ 1] = {"J", "S", 0, columns_-2};
    piece_positions_[ 2] = {"J", "E", 0, columns_-1};
    piece_positions_[ 3] = {"J", "W", 0, columns_-1};
    piece_positions_[ 4] = {"L", "N", 0, columns_-2};
    piece_positions_[ 5] = {"L", "S", 0, columns_-2};
    piece_positions_[ 6] = {"L", "E", 0, columns_-1};
    piece_positions_[ 7] = {"L", "W", 0, columns_-1};
    piece_positions_[ 8] = {"T", "N", 0, columns_-2};
    piece_positions_[ 9] = {"T", "S", 0, columns_-2};
    piece_positions_[10] = {"T", "E", 0, columns_-1};
    piece_positions_[11] = {"T", "W", 0, columns_-1};
    piece_positions_[12] = {"O", "S", 0, columns_-1};
    piece_positions_[13] = {"I", "S", 0, columns_-3};
    piece_positions_[14] = {"I", "W", 0, columns_-0};
    piece_positions_[15] = {"S", "S", 0, columns_-2};
    piece_positions_[16] = {"S", "W", 0, columns_-1};
    piece_positions_[17] = {"Z", "S", 0, columns_-1};
    piece_positions_[18] = {"Z", "W", 0, columns_-1};

    vector<vector<int>> positions[] = {
      {{0, 0}, {0, 1}, {0, 2}, {1, 2}},
      {{0, 0}, {1, 0}, {1, 1}, {1, 2}},
      {{2, 0}, {1, 1}, {2, 1}, {0, 1}},
      {{0, 0}, {1, 0}, {2, 0}, {0, 1}},
      {{0, 0}, {0, 1}, {0, 2}, {1, 0}},
      {{1, 0}, {1, 1}, {1, 2}, {-1, 1}},
      {{-1, 0}, {0, 0}, {1, 0}, {-1, -1}},
      {{-1, 0}, {0, 0}, {1, 0}, {1, 1}},
      {{0, -1}, {0, 0}, {0, 1}, {1, 0}},
      {{0, -1}, {0, 0}, {0, 1}, {-1, 0}},
      {{0, -1}, {0, 0}, {-1, 0}, {1, 0}},
      {{0, 1}, {0, 0}, {-1, 0}, {1, 0}},
      {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
      {{0, 0}, {0, 1}, {0, 2}, {0, 3}},
      {{0, 0}, {-1, 0}, {1, 0}, {2, 0}},
      {{0, 0}, {-1, 0}, {0, 1}, {1, 1}},
      {{0, -1}, {0, 0}, {-1, 0}, {-1, 1}},
      {{0, 0}, {0, 1}, {-1, 1}, {1, 0}},
      {{-1, -1}, {-1, 0}, {0, 0}, {0, 1}},
    };

    for (int i = 0; i < 19; ++i) {
      PiecePositions& pp = piece_positions_[i];
      vector<vector<int>>& p = positions[i];
      pp.setPiecePosition(p);
      pp.idx_ = piece_name_to_idx_[pp.name_];
    }

    for (int i = 0; i < 19; ++i) {
      PiecePositions& pp = piece_positions_[i];
      for (int pos_place = pp.min_x_; pos_place < pp.max_x_; ++pos_place) {
        AbsStartPiecePositions absPos;
        absPos.copyPos(pp);

        for (int j = 0; j < 4; ++j) {
          absPos.pos_[j][0] += 1;
          absPos.pos_[j][1] += pos_place;
        }

        key_name_dir_pos key = std::make_tuple(pp.name_, pp.direction_, pos_place);
        keys_.push_back(key);
        piece_name_direction_pos_to_abs_pos_[key] = absPos;
      }
    }
  }

  ~TetrisField() {
    delete[] field_;
    delete[] field_arr_;
  }

  void printCurrentPiece() {
    cout << "current_piece_.name_: " << current_piece_.name_ << endl;
    cout << "current_piece_.direction_: " << current_piece_.direction_ << endl;
    cout << "current_piece_.pos_place_: " << current_piece_.pos_place_ << endl;
  }

  uint8_t** const getField() {
    return field_;
  }
 
  void printField() {
    for (int j = 0; j < rows_; ++j) {
      uint8_t* row = field_[j];
      for (int i = 0; i < columns_; ++i) {
        cout << unsigned(row[i]) << ",";
      }
      cout << endl;
    }
  }

  friend ostream& operator<<(ostream& os, const TetrisField& obj) {
    for (int j = 0; j < obj.rows_; ++j) {
      uint8_t* row = obj.field_[j];
      os << unsigned(row[0]);
      for (int i = 1; i < obj.columns_; ++i) {
        os << "," << unsigned(row[i]);
      }
      os << "\n";
    }
    return os;
  }

  void setPiece(string piece_name, string direction, int pos_place) {
    auto key = std::make_tuple(piece_name, direction, pos_place);
    auto it = piece_name_direction_pos_to_abs_pos_.find(key);
    if(it != piece_name_direction_pos_to_abs_pos_.end())
    {
      auto absPos = it->second;

      current_piece_.is_piece_set_ = true;
      current_piece_.name_ = piece_name;
      current_piece_.direction_ = direction;
      current_piece_.key_ = key;
      current_piece_.pos_place_ = pos_place;
      current_piece_.idx_ = piece_name_to_idx_[piece_name];

      current_piece_.copyPos(absPos);
    }
  }

  void setRandomPiece() {
    key_name_dir_pos k = *select_randomly(keys_.begin(), keys_.end());
    setPiece(std::get<0>(k), std::get<1>(k), std::get<2>(k));
  }

  bool moveCurrentPieceInstant() {
    if (!current_piece_.is_piece_set_) {
      return false;
    }

    const size_t size = current_piece_.pos_.size();

    for (size_t j = 0; j < size; ++j) {
      vector<int>& point = current_piece_.pos_[j];
      if (field_[point[0]][point[1]] != 0) {
        return false;
      }
    }

    bool is_possible_to_move = true;
    while (is_possible_to_move) {
      for (size_t j = 0; j < size; ++j) {
        vector<int>& point = current_piece_.pos_[j];
        point[0] += 1;
        if ((point[0] >= rows_) || (field_[point[0]][point[1]] != 0)) {
          is_possible_to_move = false;
          
          for (int i = 0; i <= j; ++i) {
            vector<int>& point_2 = current_piece_.pos_[i];
            point_2[0] -= 1;
          }
          break;
        }
      }
    }

    for (size_t j = 0; j < size; ++j) {
      vector<int>& point = current_piece_.pos_[j];
      field_[point[0]][point[1]] = current_piece_.idx_;
    }

    current_piece_.is_piece_set_ = false;

    return true;
  }

  void clearField() {
    for (int j = 0; j < rows_; ++j) {
      uint8_t* row = field_[j];
      for (int i = 0; i < columns_; ++i) {
        row[i] = 0;
      }
    }
  }
};

int main(int argc, char* argv[]) {
  TetrisField tf = TetrisField(20, 5);

  do {
    tf.setRandomPiece();
    bool isPieceMoved = tf.moveCurrentPieceInstant();
    cout << "field:" << endl << tf;
    tf.printCurrentPiece();
    tf.clearField();
    cout << "ENTER..." << endl;
  } while (std::cin.get() == '\n');
  return 0;
}
