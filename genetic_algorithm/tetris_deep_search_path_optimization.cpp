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

class TetrisField {
private:
  uint8_t* field_arr_;
  uint8_t** field_;
  
  int rows_;
  int columns_;

  struct PiecePositions {
    string name_;
    string direction_;
    int min_x_;
    int max_x_;
    int pos_[4][2];

    void printAll() {
      ostringstream ss;
      ss << "name_: " << name_ << endl;
      string s = ss.str();
      cout << s;
    }

    void printPos() {
      ostringstream ss;
      for (int i = 0; i < 4; ++i) {
        ss << "(" << pos_[i][0] << ", " << pos_[i][1] << "), ";
      }
      string s = ss.str();
      s.resize(s.size() - 2);
      cout << "pos_: " << s << endl;
    }
  };
  PiecePositions piece_positions_[19];

  struct AbsStartPiecePositions {
    int pos_[4][2];

    void copyPos(int pos[4][2]) {
      for (int i = 0; i < 4; ++i) {
        pos_[i][0] = pos[i][0];
        pos_[i][1] = pos[i][1];
      }
    }

    void printPos() {
      ostringstream ss;
      for (int i = 0; i < 4; ++i) {
        ss << "(" << pos_[i][0] << ", " << pos_[i][1] << "), ";
      }
      string s = ss.str();
      s.resize(s.size() - 2);
      cout << "pos_: " << s << endl;
    }
  };
  
  vector<string> pieces_name_ = {"T", "O", "L", "S", "Z", "J", "I"};
  vector<uint8_t> pieces_idx_ = {1, 2, 3, 4, 5, 6, 7};
  map<string, uint8_t> piece_name_to_idx_;
  map<string, uint8_t> piece_name_to_index_;

  typedef std::tuple<string, string, int> key_name_dir_pos;
  map<key_name_dir_pos, AbsStartPiecePositions> piece_name_direction_pos_to_abs_pos_;
  vector<key_name_dir_pos> keys_;
public:
  TetrisField(const int rows, const int columns) {
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

    piece_positions_[0] = {"J", "N", 1, columns_-1, {{0, -1}, {0, 0}, {0, 1}, {1, 1}}};
    piece_positions_[1] = {"J", "S", 1, columns_-1, {{0, -1}, {0, 0}, {0, 1}, {-1, -1}}};
    piece_positions_[2] = {"J", "E", 1, columns_, {{-1, 0}, {0, 0}, {1, 0}, {1, -1}}};
    piece_positions_[3] = {"J", "W", 0, columns_-1, {{-1, 0}, {0, 0}, {1, 0}, {-1, 1}}};
    piece_positions_[4] = {"L", "N", 1, columns_-1, {{0, -1}, {0, 0}, {0, 1}, {1, -1}}};
    piece_positions_[5] = {"L", "S", 1, columns_-1, {{0, -1}, {0, 0}, {0, 1}, {-1, 1}}};
    piece_positions_[6] = {"L", "E", 1, columns_, {{-1, 0}, {0, 0}, {1, 0}, {-1, -1}}};
    piece_positions_[7] = {"L", "W", 0, columns_-1, {{-1, 0}, {0, 0}, {1, 0}, {1, 1}}};
    piece_positions_[8] = {"T", "N", 1, columns_-1, {{0, -1}, {0, 0}, {0, 1}, {1, 0}}};
    piece_positions_[9] = {"T", "S", 1, columns_-1, {{0, -1}, {0, 0}, {0, 1}, {-1, 0}}};
    piece_positions_[10] = {"T", "E", 1, columns_, {{0, -1}, {0, 0}, {-1, 0}, {1, 0}}};
    piece_positions_[11] = {"T", "W", 0, columns_-1, {{0, 1}, {0, 0}, {-1, 0}, {1, 0}}};
    piece_positions_[12] = {"O", "S", 0, columns_-1, {{0, 0}, {0, 1}, {1, 0}, {1, 1}}};
    piece_positions_[13] = {"I", "S", 1, columns_-2, {{0, 0}, {0, 1}, {0, -1}, {0, 2}}};
    piece_positions_[14] = {"I", "W", 0, columns_, {{0, 0}, {-1, 0}, {1, 0}, {2, 0}}};
    piece_positions_[15] = {"S", "S", 0, columns_-1, {{0, 0}, {-1, 0}, {0, 1}, {1, 1}}};
    piece_positions_[16] = {"S", "W", 1, columns_-1, {{0, -1}, {0, 0}, {-1, 0}, {-1, 1}}};
    piece_positions_[17] = {"Z", "S", 0, columns_-1, {{0, 0}, {0, 1}, {-1, 1}, {1, 0}}};
    piece_positions_[18] = {"Z", "W", 1, columns_-1, {{-1, -1}, {-1, 0}, {0, 0}, {0, 1}}};

    for (int i = 0; i < 19; ++i) {
      PiecePositions piPos = piece_positions_[i];
      for (int pos = piPos.min_x_; pos < piPos.max_x_; ++pos) {
        AbsStartPiecePositions absPos;
        absPos.copyPos(piPos.pos_);
        for (int j = 0; j < 4; ++j) {
          absPos.pos_[j][0] += 1;
          absPos.pos_[j][1] += pos;
        }
        // absPos.printPos();
        key_name_dir_pos key = std::make_tuple(piPos.name_, piPos.direction_, pos);
        keys_.push_back(key);
        piece_name_direction_pos_to_abs_pos_[key] = absPos;
      }
    }
  }

  ~TetrisField() {
    delete[] field_;
    delete[] field_arr_;
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

  void setPiece(string piece_name, string direction, int pos) {
    auto it = piece_name_direction_pos_to_abs_pos_.find(std::make_tuple(piece_name, direction, pos));
    if(it != piece_name_direction_pos_to_abs_pos_.end())
    {
      auto absPos = it->second;
      // absPos.printPos();

      auto it_idx = piece_name_to_idx_.find(piece_name);
      auto pcs_idx = it_idx->second;
      auto pos = absPos.pos_;
      for (int i = 0; i < 4; ++i) {
        field_[pos[i][0]][pos[i][1]] = pcs_idx;
      }
    }
  }

  void setRandomPiece() {
    key_name_dir_pos k = *select_randomly(keys_.begin(), keys_.end());
    setPiece(std::get<0>(k), std::get<1>(k), std::get<2>(k));
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

  while (std::cin.get() == '\n') {
    tf.setRandomPiece();
    cout << "field:" << endl << tf;
    tf.clearField();
  }
  return 0;
}
