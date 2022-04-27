#include "utils.h"
#include "RandomGen.h"

#include "CImg.h"
#define cimg_use_png

using RandomGen::RandomGenInt;

template<typename T>
class Matrix {
public:
	vector<T> arr_;
	size_t rows_;
	size_t cols_;

	size_t amount_;

	Matrix() {
		arr_.resize(0);
		rows_ = 0;
		cols_ = 0;
		amount_ = 0;
	}

	Matrix(const size_t rows, const size_t cols, const size_t default_val) {
		rows_ = rows;
		cols_ = cols;
		amount_ = rows * cols;
		arr_.resize(amount_);
		fill(arr_.begin(), arr_.end(), default_val);
	}

	Matrix& operator=(const Matrix&) = default;

	void saveMatrixToBinary(const path& file_path) {
		fstream f_out(file_path, std::ios::out | std::ios::binary);
		if(!f_out) {
			cout << "Cannot open file!" << endl;
			return;
		}

		const uint8_t dim = 2;
		f_out.write((char*)&dim, sizeof(dim));

		uint64_t val;
		val = rows_;
		f_out.write((char*)&val, sizeof(val));
		val = cols_;
		f_out.write((char*)&val, sizeof(val));

		val = amount_;
		f_out.write((char*)&val, sizeof(val));

		f_out.write((char*)&arr_[0], sizeof(T) * amount_);

		f_out.close();
	}
};

using MatType = uint8_t;
class Sequence2d {
public:
	Matrix<MatType> mat_;
	uint64_t modulo_;
	uint64_t v_const_;
	uint64_t v_a_;
	uint64_t v_b_;
	uint64_t v_c_;

	Sequence2d(const size_t rows, const size_t cols, const uint64_t modulo) {
		mat_ = Matrix<MatType>(rows, cols, 0);
		modulo_ = modulo;
	}

	inline void initZeroMatrix() {
		fill(mat_.arr_.begin(), mat_.arr_.end(), 0);
	}

	inline void initModuloFirstRowFirstColumn() {
		vector<MatType>& vec = mat_.arr_;

		const size_t rows = mat_.rows_;
		const size_t cols = mat_.cols_;

		{
			uint64_t temp_val = 0;
			for (size_t x = 0; x < cols; ++x) {
				vec[x] = temp_val;
				temp_val = (temp_val + 1) % modulo_;
			}
		}

		{
			uint64_t temp_val = 0;
			for (size_t y = 0; y < rows; ++y) {
				vec[y * cols] = temp_val;
				temp_val = (temp_val + 1) % modulo_;
			}
		}
	}

	void sequenceNr1(const uint64_t v_const, const uint64_t v_a, const uint64_t v_b, const uint64_t v_c) {
		v_const_ = v_const;
		v_a_ = v_a;
		v_b_ = v_b;
		v_c_ = v_c;

		const size_t rows = mat_.rows_;
		const size_t cols = mat_.cols_;

		initZeroMatrix();
		initModuloFirstRowFirstColumn();

		vector<MatType>& vec = mat_.arr_;
		for (size_t y = 1; y < rows; ++y) {
			for (size_t x = 1; x < cols; ++x) {
				const size_t i = (y - 0) * cols + (x - 0);
				const size_t i_1 = (y - 1) * cols + (x - 0);
				const size_t i_2 = (y - 0) * cols + (x - 1);
				const size_t i_3 = (y - 1) * cols + (x - 1);

				assert(i < mat_.amount_);
				assert(i_1 < mat_.amount_);
				assert(i_2 < mat_.amount_);
				assert(i_3 < mat_.amount_);

				uint64_t val = v_const;
				val = (val + (uint64_t)vec[i_1] * v_a) % modulo_;
				val = (val + (uint64_t)vec[i_2] * v_b) % modulo_;
				val = (val + (uint64_t)vec[i_3] * v_c) % modulo_;

				vec[i] = val;
			}
		}
	}

	void saveToBinary(const path& file_path) {
		fstream f_out(file_path, std::ios::out | std::ios::binary);
		if(!f_out) {
			cout << "Cannot open file!" << endl;
			return;
		}

		uint64_t val;
		val = modulo_;
		f_out.write((char*)&val, sizeof(val));
		val = v_const_;
		f_out.write((char*)&val, sizeof(val));
		val = v_a_;
		f_out.write((char*)&val, sizeof(val));
		val = v_b_;
		f_out.write((char*)&val, sizeof(val));
		val = v_c_;
		f_out.write((char*)&val, sizeof(val));

		const uint8_t dim = 2;
		f_out.write((char*)&dim, sizeof(dim));

		val = mat_.rows_;
		f_out.write((char*)&val, sizeof(val));
		val = mat_.cols_;
		f_out.write((char*)&val, sizeof(val));

		val = mat_.amount_;
		f_out.write((char*)&val, sizeof(val));

		f_out.write((char*)&mat_.arr_[0], sizeof(MatType) * mat_.amount_);

		f_out.close();
	}
};

int main(const int argc, const char* argv[]) {
  const path dir_path = path("/tmp/generate_2d_sequence");

  if (!fs::is_directory(dir_path) || !fs::exists(dir_path)) {
    fs::create_directory(dir_path);
  }

	const int32_t height = std::stoi(argv[1]);
	const int32_t width = std::stoi(argv[2]);
	const uint32_t modulo = std::stoi(argv[3]);
	const uint32_t v_const = std::stoi(argv[4]);
	const uint32_t v_a = std::stoi(argv[5]);
	const uint32_t v_b = std::stoi(argv[6]);
	const uint32_t v_c = std::stoi(argv[7]);
	
	const int32_t color_depth = 3;
	cimg_library::CImg<uint8_t> img(width, height, 1, color_depth, 0);

	Sequence2d sequence_2d = Sequence2d(height, width, modulo);

	vector<vector<uint8_t>> all_colors;
	all_colors.resize(modulo);

	const double spaces = 256. / (modulo - 1);
	for (size_t i = 0; i < modulo - 1; ++i) {
		vector<uint8_t>& vec = all_colors[i];
		vec.resize(color_depth);

		const uint8_t val = spaces * i;
		for (size_t j = 0; j < 3; ++j) {
			vec[j] = val;
		}
	}

	{
		vector<uint8_t>& vec = all_colors[modulo - 1];
		vec.resize(color_depth);
		const uint8_t val = 255;
		for (size_t j = 0; j < 3; ++j) {
			vec[j] = val;
		}
	}

	// cout << format("all_colors: {}", all_colors) << endl;

	// for (uint32_t v_a = 0; v_a < modulo; ++v_a) {
	// 	for (uint32_t v_b = 0; v_b < modulo; ++v_b) {
	// 		for (uint32_t v_c = 0; v_c < modulo; ++v_c) {
	// 			for (uint32_t v_const = 0; v_const < modulo; ++v_const) {
					sequence_2d.sequenceNr1(v_const, v_a, v_b, v_c);

					// TODO: combine the sequence with the creation of the image too! next step: add generic creating of pixel values per modulo!

					std::vector<uint32_t> seed = {0, 1, 2};
					RandomGenInt<uint8_t> rng(seed);

					for (int32_t y = 0; y < height; ++y) {
						for (int32_t x = 0; x < width; ++x) {
							const size_t idx = y * width + x;
							const vector<uint8_t>& colors = all_colors[sequence_2d.mat_.arr_[idx]];
							img.draw_point(x, y, (uint8_t*)&colors[0]);
						}
					}

					const string base_name = format("m_{:04}_v_a_{:02}_v_b_{:02}_v_c_{:02}_v_const_{:02}", modulo, v_a, v_b, v_c, v_const);
					const path file_path = dir_path / format("{}.png", base_name);
					img.save(file_path.string().c_str());

					// sequence_2d.saveToBinary(dir_path / format("{}.data", base_name));
	// 			}	
	// 		}	
	// 	}	
	// }

  // for (uint32_t y = 0; y < height; ++y) {
  // 	for (uint32_t x = 0; x < width; ++x) {
  // 		cout << format("{},", sequence_2d.mat_.arr_[y*width+x]);
  // 	}
  // 	cout << endl;
  // }

	return 0;
}
