#include <iostream>
#include <utility>
#include <vector>
#include <cstdint>
#include <ostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <set>
#include <tuple>
#include <exception>
#include <ios>
#include <unistd.h>
#include <filesystem>
#include <variant>
#include "pystring/pystring.h"

#include "utils_tetris.h"

#include "../cpp_programs/primes/utils.h"

using std::cout;
using std::cin;
using std::endl;
using std::vector;
using std::fill;
using std::ostream;
using std::ifstream;
using std::stringstream;
using std::ostringstream;
using std::string;
using std::map;
using std::set;
using std::tuple;
using std::variant;

namespace fs = std::filesystem;

class TetrisField : public std::exception {
private:
  string data_path_;
  string file_data_name_;

  vector<vector<uint8_t>> field_;
  vector<vector<int>> field_weight_;

  int rows_;
  int columns_;

  typedef std::tuple<string, string, int> key_name_dir_pos;

  struct Position {
    int rows_{};
    int columns_{};
    vector<vector<int>> pos_;

    void copyPos(const int rows, const int columns, const uint8_t *pos) {
      rows_ = rows;
      columns_ = columns;
      pos_.resize(rows);
      for (int j = 0; j < rows; ++j) {
        vector<int>& v = pos_[j];
        v.resize(columns);
        for (int i = 0; i < columns; ++i) {
          v[i] = pos[j * columns + i];
        }
      }
    }
  };

  struct PiecePositions;

  struct Piece : public Position {
    string name_;
    string direction_;
    int pos_place_;
    int field_idx_;
    int piece_unq_idx_;
    int piece_idx_;

    Piece(const Piece& obj) {
      copyValues(obj);
    }

    Piece(const PiecePositions& pp, const int pos_place) {
      rows_ = pp.rows_;
      columns_ = pp.columns_;
      pos_ = pp.pos_;

      const size_t size = pos_.size();
      for (size_t i = 0; i < size; ++i) {
        pos_[i][1] += pos_place;
      }

      name_ = pp.name_;
      direction_ = pp.direction_;
      pos_place_ = pos_place;
      piece_unq_idx_ = pp.piece_unq_idx_;
      field_idx_ = piece_unq_idx_ + 1;
      piece_idx_ = pp.piece_idx_;
    }

    friend ostream& operator<<(ostream& os, Piece const& obj) {
      os << "name_: " << obj.name_;
      os << ", piece_unq_idx_: " << obj.piece_unq_idx_;
      os << ", field_idx_: " << obj.field_idx_;
      os << ", piece_idx_: " << obj.piece_idx_;
      os << ", direction_: " << obj.direction_;
      os << ", pos_place_: " << obj.pos_place_;
      os << ", pos_: " << obj.pos_;
      return os;
    }

    void copyValues(const Piece& obj) {
      name_ = obj.name_;
      direction_ = obj.direction_;
      pos_place_ = obj.pos_place_;
      piece_unq_idx_ = obj.piece_unq_idx_;
      field_idx_ = obj.piece_unq_idx_ + 1;
      piece_idx_ = obj.piece_idx_;

      rows_ = obj.rows_;
      columns_ = obj.columns_;
      pos_ = obj.pos_;
    }
  };

  struct PiecePositions : public Position {
    string name_;
    string direction_;
    int min_x_;
    int max_x_;
    int field_idx_;
    int piece_unq_idx_;
    int piece_idx_;

    PiecePositions(
      string name, string direction, const int rows, const int columns, const uint8_t *pos,
      const int field_columns, const int piece_unq_idx, const int piece_idx
    ) :
      name_(std::move(name)), direction_(std::move(direction)), field_idx_(piece_unq_idx + 1), piece_unq_idx_(piece_unq_idx), piece_idx_(piece_idx) {
      copyPos(rows, columns, pos);

      int min_x = pos_[0][1];
      int max_x = pos_[0][1];

      for (int j = 1; j < rows; ++j) {
        const vector<int>& v = pos_[j];
        const int x = v[1];

        if (min_x > x) {
          min_x = x;
        }
        if (max_x < x) {
          max_x = x;
        }
      }

      min_x_ = -min_x;
      max_x_ = field_columns - max_x;
    }
  };

  vector<PiecePositions> piece_positions_;
  vector<uint8_t> group_piece_amount_;

  set<string> set_pieces_name_;
  vector<string> vec_pieces_name_;
  int piece_unq_idx_;
  int piece_idx_;

  map<string, string> piece_name_to_piece_orientation_;
//  map<string, uint8_t> piece_name_to_index_;
  map<string, uint8_t> piece_name_to_field_idx_;
  map<string, uint8_t> piece_name_to_piece_unq_idx_;

  int amount_pieces_sequence;
  vector<string> pieces_sequence_;

  map<string, vector<Piece>> piece_name_to_group_pieces_;

  void readTetrisData(const string& file_path) {
    std::ifstream stream(file_path, std::ios::in | std::ios::binary);
    std::vector<uint8_t> contents((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());

    const uint8_t* data = contents.data();

    const uint8_t num_blocks = data[0];
    const uint8_t amount_groups = data[1];

    const uint8_t* group_piece_amount = data + 2;

    const uint8_t* pos_data_start = (uint8_t *) data + (2 + (int) amount_groups);

    cout << "num_blocks: " << unsigned(num_blocks) << endl;
    cout << "amount_groups: " << unsigned(amount_groups) << endl;

    int index_orientation = 0;

    vector<string> pieces_name;
    vector<string> pieces_orientation;
    // vector<int> pieces_idx_nr;

//    if (num_blocks == 4) {
//      pieces_name = {"O", "I", "Z", "S", "L", "T", "J"};
//      pieces_orientation = {
//        "S", "S", "W", "S", "W", "W", "S", "N", "E", "W",
//        "S", "N", "W", "S", "E", "N", "W", "S", "E",
//      };
//    } else {
    pieces_name.resize(amount_groups);

    int sum = 0;
    for (int i = 0; i < amount_groups; ++i) {
      const uint8_t& val = group_piece_amount[i];
      sum += (int) val;
      group_piece_amount_.push_back(val);
      pieces_name[i] = std::to_string(i + 1);
    }
    pieces_orientation.resize(sum);

    int idx = 0;
    for (int j = 0; j < amount_groups; ++j) {
      const int amount = (int) group_piece_amount[j];
      if (amount == 1) {
        pieces_orientation[idx++] = "S";
      } else if (amount == 2) {
        pieces_orientation[idx++] = "S";
        pieces_orientation[idx++] = "W";
      } else if (amount == 4) {
        pieces_orientation[idx++] = "S";
        pieces_orientation[idx++] = "W";
        pieces_orientation[idx++] = "N";
        pieces_orientation[idx++] = "E";
      }
    }
//    }

    for (int i = 0; i < amount_groups; ++i) {
      pieces_name[i] = std::to_string(unsigned(num_blocks)) + "_" + pieces_name[i];
    }

    map<string, int> piece_name_to_index;

    for (int i = 0; i < amount_groups; ++i) {
      piece_name_to_index[pieces_name[i]] = i;
    }

    uint8_t *pos_data_next = (uint8_t *) pos_data_start;
    for (int i_group = 0; i_group < amount_groups; ++i_group) {
      const uint8_t piece_amount = group_piece_amount[i_group];
      const string name = pieces_name[i_group];
      const string orientation = pieces_orientation[i_group];

      if (set_pieces_name_.find(name) != set_pieces_name_.end()) {
        continue;
      }

      set_pieces_name_.insert(name);
      vec_pieces_name_.push_back(name);
//      piece_name_to_field_idx_[name] = vec_pieces_name_.size();
//      piece_name_to_index_[name] = piece_name_to_index[name];
      piece_name_to_piece_orientation_[name] = orientation;

      piece_name_to_piece_unq_idx_[name] = piece_unq_idx_;
      piece_name_to_field_idx_[name] = piece_unq_idx_ + 1;

      piece_name_to_group_pieces_[name] = {};
      vector<Piece>& group_pieces = piece_name_to_group_pieces_[name];

      for (int i_piece = 0; i_piece < piece_amount; ++i_piece) {
        piece_positions_.emplace_back(name, pieces_orientation[index_orientation], num_blocks, 2,
                                      pos_data_next, columns_, piece_unq_idx_, piece_idx_);
        ++piece_idx_;
        PiecePositions& pp = piece_positions_.back();

        for (int pos_place = pp.min_x_; pos_place < pp.max_x_; ++pos_place) {
          group_pieces.emplace_back(pp, pos_place);
        }

        ++index_orientation;
        pos_data_next += num_blocks * 2;
      }

      ++piece_unq_idx_;

      cout << "name: " << name;
      cout << "; group_pieces.size(): " << group_pieces.size() << endl;
    }
  }

public:
  TetrisField(const int rows, const int columns, const vector<int>& vec_num_blocks = {4},
      const string& data_path = "tetris_game_data/", const string& file_data_name = "data_fields_test",
      const string& file_extension = "ttrsfields") {
    piece_unq_idx_ = 0;
    piece_idx_ = 0;

    data_path_ = data_path;
    file_data_name_ = file_data_name + "." + file_extension;

    rows_ = rows;
    columns_ = columns;

    field_.resize(rows);
    for (int i = 0; i < rows; ++i) {
      field_[i].resize(columns, 0);
    }

    int sum_row_acc = 1;
    int sum_row = 1;
    field_weight_.resize(rows);
    for (int j = rows - 1; j > -1; --j) {
      vector<int>& v = field_weight_[j];
      v.resize(columns);
      for (int i = 0; i < columns; ++i) {
        v[i] = i + sum_row;
      }
      sum_row_acc += 1;
      sum_row += sum_row_acc;
    }

    set<int> done_num_blocks;
    for (int num_blocks : vec_num_blocks) {
      if (done_num_blocks.find(num_blocks) != done_num_blocks.end()) {
        cout << "num_blocks: " << num_blocks << " is already finished!" << endl;
        continue;
      }
      done_num_blocks.insert(num_blocks);

      cout << "Doing now: num_blocks: " << num_blocks << endl;

      stringstream ss;
      ss << "tetris_data/tetris_pieces_block_amount_" << num_blocks << ".trpcs";

      const string file_path = ss.str();
      cout << "file_path: " << file_path << endl;

      readTetrisData(file_path);
    }
  }

  void printField() {
    for (int j = 0; j < rows_; ++j) {
      vector<uint8_t>& v = field_[j];
      for (int i = 0; i < columns_; ++i) {
        cout << unsigned(v[i]) << ",";
      }
      cout << endl;
    }
  }

  friend ostream& operator<<(ostream& os, const TetrisField& obj) {
    for (int j = 0; j < obj.rows_; ++j) {
      const vector<uint8_t>& v = obj.field_[j];
      os << unsigned(v[0]);
      for (int i = 1; i < obj.columns_; ++i) {
        os << "," << unsigned(v[i]);
      }
      os << "\n";
    }
    return os;
  }

  stringstream& printKey(stringstream& ss, const key_name_dir_pos& obj) {
    ss << "name: " << std::get<0>(obj) << ", direction: " << std::get<1>(obj) << ", pos_place: " << std::get<2>(obj);
    return ss;
  }

  void defineRandomPieceVector(const int amount_pieces_sequence = 100) {
    pieces_sequence_.resize(amount_pieces_sequence);

    auto get_random_piece = [&]() {
        return *select_randomly(set_pieces_name_.begin(), set_pieces_name_.end());
    };

    pieces_sequence_[0] = get_random_piece();
    pieces_sequence_[1] = get_random_piece();
    string piece_name;
    for (int i = 2; i < amount_pieces_sequence; ++i) {
      do {
        piece_name = get_random_piece();
      } while (
        (pieces_sequence_[i - 2] == piece_name) &&
        (pieces_sequence_[i - 1] == piece_name)
        );
      pieces_sequence_[i] = piece_name;
    }

    cout << "pieces_sequence_: " << pieces_sequence_ << endl;
  }

  int returnMaxHeight() {
    for (int row = 0; row < rows_; ++row) {
      bool is_one_cell_set = false;
      for (int column = 0; column < columns_; ++column) {
        if (field_[row][column]) {
          is_one_cell_set = true;
          break;
        }
      }

      if (is_one_cell_set) {
        return rows_ - row;
      }
    }

    return 0;
  }

  int calculateWeightSum() {
    int sum = 0;

    for (int j = 0; j < rows_; ++j) {
      for (int i = 0; i < columns_; ++i) {
        sum += (field_[j][i] ? 1 : 0) * field_weight_[j][i];
      }
    }

    return sum;
  }

  void executePieceVector(const int using_pieces = 3) {
    vector<uint8_t> heights;
    vector<vector<uint8_t>> relative_row_heights_per_column;
    vector<int> piece_group_idx(using_pieces);

    vector<uint8_t> fields;

    struct IdxPosPlace {
      uint8_t piece_unq_idx_;
      uint8_t piece_idx_;
      uint8_t pos_place_;

      IdxPosPlace(const uint8_t piece_unq_idx, const uint8_t piece_idx, const uint8_t pos_place) {
        piece_unq_idx_ = piece_unq_idx;
        piece_idx_ = piece_idx;
        pos_place_ = pos_place;
      }
    };
    vector<IdxPosPlace> used_piece_idx_pos;

    auto copy_field_to_data_fields = [&]() {
      for (size_t j = 0; j < field_.size(); ++j) {
        vector<uint8_t>& v = field_[j];
        fields.reserve(fields.size() + v.size());
        fields.insert(fields.end(), v.begin(), v.end());
      }
    };

    auto save_the_heights_of_rows_relative_per_column = [&]() {
      // first find the first appearance of the max height per column
      set<uint8_t> col_idx_not_finished;
      vector<uint8_t> col_max_height(columns_, rows_);

      for (int i = 0; i < columns_; ++i) {
        col_idx_not_finished.emplace(i);
      }

      uint8_t min_height = 0;
      for (int row = 0; (row < rows_) && (!col_idx_not_finished.empty()); ++row) {
        vector<uint8_t>& v = field_[row];

        for(set<uint8_t>::iterator iter = col_idx_not_finished.begin(); iter != col_idx_not_finished.end();) {
          if(v[*iter]) {
            col_max_height[*iter] = row;
            if (col_idx_not_finished.size() == 1) {
              min_height = rows_ - row;
            }
            iter = col_idx_not_finished.erase(iter);
          } else {
            ++iter;
          }
        }
      }

      for (vector<uint8_t>::iterator it = col_max_height.begin(); it != col_max_height.end(); ++it) {
        *it = rows_ - *it - min_height;
      }

      relative_row_heights_per_column.emplace_back(col_max_height);
    };

    struct Datas {
      int height_;
      int weight_;
      int lines_removed_;

      void print(const string& name = "min_data") const {
        cout << name << ": " << "height_: " << height_ << ", weight_: " << weight_ << ", lines_removed_: " << lines_removed_ << endl;
      }
    };

    // is needed to get in sync with the next two times copy_field is called
    copy_field_to_data_fields();
    copy_field_to_data_fields();

    const size_t size = pieces_sequence_.size() - using_pieces + 1;
    for (size_t i = 0; i < size; ++i) {
      map<vector<int>, Datas> pos_pieces_to_max_height;
      vector<string> using_pieces_sequence(pieces_sequence_.begin() + i, pieces_sequence_.begin() + i + using_pieces);

      std::function<void(int, int, int, int)> do_resursive;
      do_resursive = [&](const int idx, const int idx_maxs, const int piece_idx, const int lines_removed_prev) {
          vector<Piece>& vp = piece_name_to_group_pieces_[pieces_sequence_[idx]];
          const size_t size = vp.size();
          for (size_t i_vp = 0; i_vp < size; ++i_vp) {
            piece_group_idx[piece_idx] = i_vp;
            Piece p(vp[i_vp]);
            vector<vector<uint8_t>> field_copy(field_);
            movePieceInstant(p);
            int lines_removed = removeFullLines();

            if (idx == idx_maxs) {
              const int max_height = returnMaxHeight();
              const int lines_removed_total = lines_removed_prev + lines_removed;
              pos_pieces_to_max_height[piece_group_idx] = {max_height, calculateWeightSum(), lines_removed_total};
            } else {
              do_resursive(idx + 1, idx_maxs, piece_idx + 1, lines_removed_prev + lines_removed);
            }

            field_ = field_copy;
          }
      };
      do_resursive(i, i + using_pieces - 1, 0, 0);

      cout << "i: " << i << endl;
      const vector<int> *best_piece_group_idx = &(pos_pieces_to_max_height.begin()->first);
      Datas min_data = pos_pieces_to_max_height.begin()->second;
      for (auto& it : pos_pieces_to_max_height) {
        const vector<int>& pos_pieces = it.first;
        Datas& datas = it.second;

        if (min_data.lines_removed_ < datas.lines_removed_) {
          min_data = datas;
          best_piece_group_idx = &pos_pieces;
        } else if (min_data.lines_removed_ == datas.lines_removed_) {
          if (min_data.height_ > datas.height_) {
            min_data = datas;
            best_piece_group_idx = &pos_pieces;
          } else if (min_data.height_ == datas.height_) {
            if (min_data.weight_ > datas.weight_) {
              min_data = datas;
              best_piece_group_idx = &pos_pieces;
            }
          }
        }
      }

      vector<Piece>& vp = piece_name_to_group_pieces_[pieces_sequence_[i]];
      Piece& p_ref = vp[(*best_piece_group_idx)[0]];

      used_piece_idx_pos.emplace_back(p_ref.piece_unq_idx_, p_ref.piece_idx_, p_ref.pos_place_);

      Piece p(p_ref);
      movePieceInstant(p);

      copy_field_to_data_fields();

      int lines_removed = removeFullLines();

      // printField();
      const int current_height = returnMaxHeight();
      // save the relative heights of the columns relative to the min height to all columns!

      heights.push_back(current_height);
      min_data.print("min_data");
      cout << "current_height: " << current_height << endl;
      cout << "using_pieces_sequence: " << using_pieces_sequence << endl;
      cout << "best_piece_group_idx: " << (*best_piece_group_idx) << endl;
      cout << "p: " << p << endl;
      cout << "lines_removed: " << lines_removed << endl;

      copy_field_to_data_fields();
      save_the_heights_of_rows_relative_per_column();
    }

    printField();

    cout << "heights: " << heights << endl;

    vector<uint8_t> data_fields;
    data_fields.push_back(rows_);
    data_fields.push_back(columns_);

    // save the tetris pieces first! to know, which pieces where used and which not!
//    data_fields.push_back(piece_name_to_group_pieces_.size());
    data_fields.push_back(vec_pieces_name_.size());

    data_fields.reserve(data_fields.size() + group_piece_amount_.size());
    data_fields.insert(data_fields.end(), group_piece_amount_.begin(), group_piece_amount_.end());

    for (const auto& pp : piece_positions_) {
      vector<uint8_t> data_pos;
      data_pos.push_back(pp.pos_.size());
      for (const auto& v : pp.pos_) {
        data_pos.reserve(data_pos.size() + v.size());
        data_pos.insert(data_pos.end(), v.begin(), v.end());
      }
      data_fields.reserve(data_fields.size() + data_pos.size());
      data_fields.insert(data_fields.end(), data_pos.begin(), data_pos.end());
    }

    const uint16_t amount_sequence = heights.size();
    const uint8_t* amount_sequence_ptr = (uint8_t*)&amount_sequence;
    data_fields.reserve(data_fields.size() + sizeof(amount_sequence));
    data_fields.insert(data_fields.end(), amount_sequence_ptr, amount_sequence_ptr + sizeof(amount_sequence));

    for (const auto& idx_pos : used_piece_idx_pos) {
      data_fields.reserve(data_fields.size() + sizeof(IdxPosPlace));
      const auto* d = reinterpret_cast<const uint8_t*>(&idx_pos);
      data_fields.insert(data_fields.end(), d, d + sizeof(IdxPosPlace));
    }

    for (const auto& v : relative_row_heights_per_column) {
      data_fields.reserve(data_fields.size() + v.size());
      data_fields.insert(data_fields.end(), v.begin(), v.end());
    }

    data_fields.reserve(data_fields.size() + fields.size());
    data_fields.insert(data_fields.end(), fields.begin(), fields.end());

    std::ofstream stream;
    stream.open(data_path_ + file_data_name_, std::ios::out | std::ios::binary);
    stream.write((char*)&data_fields[0], data_fields.size());
  }

  bool movePieceInstant(Piece& piece) {
    const size_t size = piece.pos_.size();

    for (size_t j = 0; j < size; ++j) {
      vector<int>& point = piece.pos_[j];
      if (field_[point[0]][point[1]] != 0) {
        return false;
      }
    }

    bool is_possible_to_move = true;
    while (is_possible_to_move) {
      for (size_t j = 0; j < size; ++j) {
        vector<int>& point = piece.pos_[j];
        point[0] += 1;
        if ((point[0] >= rows_) || (field_[point[0]][point[1]] != 0)) {
          is_possible_to_move = false;

          for (size_t i = 0; i <= j; ++i) {
            vector<int>& point_2 = piece.pos_[i];
            point_2[0] -= 1;
          }
          break;
        }
      }
    }

    for (size_t j = 0; j < size; ++j) {
      vector<int>& point = piece.pos_[j];
      field_[point[0]][point[1]] = piece.field_idx_;
    }

    return true;
  }

  int removeFullLines() {
    int removed_lines = 0;
    for (int j = rows_ - 1; j > -1; --j) {
      vector<uint8_t>& row = field_[j];
      bool is_line_full = true;
      for (int i = 0; i < columns_; ++i) {
        if (!row[i]) {
          is_line_full = false;
          break;
        }
      }

      if (is_line_full) {
        for (int i = 0; i < columns_; ++i) {
          row[i] = 0;
        }
        ++removed_lines;
      } else {
        vector<uint8_t>& row_prev = field_[j + removed_lines];
        for (int i = 0; i < columns_; ++i) {
          row_prev[i] = row[i];
        }
      }
    }

    return removed_lines;
  }

  void clearField() {
    for (auto& v : field_) {
      std::fill(v.begin(), v.end(), 0);
    }
  }
};

void splitCommaStringToVector(const string& str, vector<int>& cont, const string& delim = ",") {
  vector<string> v;
  pystring::split(str, v, delim);
  const size_t size = v.size();
  cont.resize(size);
  for (size_t i = 0; i < size; ++i) {
    cont[i] = std::stoi(v[i]);
  }
  // cout << "v: " << v << endl;
}

int main(int argc, char *argv[]) {
  string argv0 = argv[0];

  if ((argv0.size() > 2) && (argv0.substr(0, 2) == "./")) {
    argv0 = argv0.substr(2, argv0.size());
  }

  fs::path p = fs::current_path();
  fs::path p_argv0 = fs::path(argv0);
  // fs::path p = prepend_exe_path("", argv0);
  cout << "p: " << p << endl;
  cout << "p_argv0: " << p_argv0 << endl;

  p /= p_argv0;
  cout << "new: p: " << p << endl;

  string path_new = p.string();
  const int pos = path_new.rfind("/");
  if (pos > 0) {
    path_new = path_new.substr(0, pos);
  }

  const int pos2 = path_new.rfind("/");
  if (pos2 > 0) {
    path_new = path_new.substr(0, pos2);
  }

  path_new += "/tetris_game_data/";

  if (!fs::exists(path_new)) {
    fs::create_directories(path_new);
  }

  cout << "path_new: " << path_new << endl;

  const string data_path = path_new;

  map<string, variant<int, string, vector<int>>> arguments_value = {
    {"-r", 20},
    {"-c", 10},
    {"-a", 100},
    {"-u", 3},
    {"-f", "data_fields_test"},
    {"-e", "ttrsfields"},
    {"-b", vector<int>({4})},
  };

  map<string, string> arguments_type = {
    {"-r", "int"},
    {"-c", "int"},
    {"-a", "int"},
    {"-u", "int"},
    {"-f", "string"},
    {"-e", "string"},
    {"-b", "vector"},
  };

  // parse the input!
  for (int i = 1; i < argc; ++i) {
    const string arg = argv[i];
    auto it = arguments_value.find(arg);
    if (it != arguments_value.end()) {
      ++i;
      if (i >= argc) {
        cout << "No more args for parameter '" << arg << "' !!! Exit program." << endl;
        exit(1);
      }

      const string arg_type = arguments_type[arg];
      if (arg_type == "int") {
        arguments_value[arg] = std::stoi(argv[i]);
      } else if (arg_type == "string") {
        arguments_value[arg] = argv[i];
      } else if (arg_type == "vector") {
        vector<int> cont;
        splitCommaStringToVector(argv[i], cont);
        arguments_value[arg] = cont;
      }
    } else {
      cout << "arg: '" << arg << "' Was not found! Exit program." << endl;
      exit(2);
    }
  }

  const int rows = std::get<int>(arguments_value["-r"]);
  const int columns = std::get<int>(arguments_value["-c"]);
  const int amount_pieces_sequence = std::get<int>(arguments_value["-a"]);
  const int using_pieces = std::get<int>(arguments_value["-u"]);
  const string& file_data_name = std::get<string>(arguments_value["-f"]);
  const string& file_extension = std::get<string>(arguments_value["-e"]);
  const vector<int>& vec_num_blocks = std::get<vector<int>>(arguments_value["-b"]);

  cout << "rows: " << rows << endl;
  cout << "columns: " << columns << endl;
  cout << "amount_pieces_sequence: " << amount_pieces_sequence << endl;
  cout << "using_pieces: " << using_pieces << endl;
  cout << "file_data_name: " << file_data_name << endl;
  cout << "file_extension: " << file_extension << endl;
  cout << "vec_num_blocks: " << vec_num_blocks << endl;

  // exit(0);

  TetrisField tf = TetrisField(rows, columns, vec_num_blocks, data_path, file_data_name, file_extension);

  tf.defineRandomPieceVector(amount_pieces_sequence + using_pieces - 1);
  tf.executePieceVector(using_pieces);

  return 0;
}
