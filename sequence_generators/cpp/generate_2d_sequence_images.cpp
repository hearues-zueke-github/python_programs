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

	Matrix(const size_t rows, const size_t cols, const size_t default_val) {
		rows_ = rows;
		cols_ = cols;
		amount_ = rows * cols;
		arr_.resize(amount_);
		fill(arr_.begin(), arr_.end(), default_val);
	}
};

class Sequence2d {
public:
	Matrix<uint32_t> mat_;
	uint32_t modulo_;

	Sequence2d(const size_t rows, const size_t cols, const uint32_t modulo) : mat_(rows, cols, 0) {
		modulo_ = modulo;
	}

	inline void initZeroMatrix() {
		fill(mat_.arr_.begin(), mat_.arr_.end(), 0);
	}

	inline void initModuloFirstRowFirstColumn() {
		vector<uint32_t>& vec = mat_.arr_;

		const size_t rows = mat_.rows_;
		const size_t cols = mat_.cols_;

		{
			uint32_t temp_val = 0;
			for (size_t x = 0; x < cols; ++x) {
				vec[x] = temp_val;
				temp_val = (temp_val + 1) % modulo_;
			}
		}

		{
			uint32_t temp_val = 0;
			for (size_t y = 0; y < cols; ++y) {
				vec[y * cols] = temp_val;
				temp_val = (temp_val + 1) % modulo_;
			}
		}
	}

	void sequenceNr1(const uint32_t v_a, const uint32_t v_b, const uint32_t v_c, const uint32_t v_const) {
		const size_t rows = mat_.rows_;
		const size_t cols = mat_.cols_;

		initZeroMatrix();
		initModuloFirstRowFirstColumn();

		vector<uint32_t>& vec = mat_.arr_;
		for (size_t y = 1; y < rows; ++y) {
			for (size_t x = 1; x < cols; ++x) {
				const size_t i = (y - 0) * cols + (x - 0);
				const size_t i_1 = (y - 0) * cols + (x - 1);
				const size_t i_2 = (y - 1) * cols + (x - 0);
				const size_t i_3 = (y - 1) * cols + (x - 1);

				uint32_t val = v_const;
				val = (val + i_1 * v_a) % modulo_;
				val = (val + i_2 * v_b) % modulo_;
				val = (val + i_3 * v_c) % modulo_;

				vec[i] = val;
			}
		}
	}
};

int main(const int argc, const char* argv[]) {
	const int32_t width = 100;
	const int32_t height = 100;
	const int32_t color_depth = 3;
	cimg_library::CImg<uint8_t> img(width, height, 1, color_depth, 0);

	Sequence2d sequence_2d = Sequence2d(100, 100, 5);
	sequence_2d.sequenceNr1(1, 1, 1, 0);

	// TODO: combine the sequence with the creation of the image too! next step: add generic creating of pixel values per modulo!

	std::vector<uint32_t> seed = {0, 1, 2};
	RandomGenInt<uint8_t> rng(seed);
	uint8_t colors[3];

	for (int32_t y = 0; y < height; ++y) {
		for (int32_t x = 0; x < width; ++x) {
			for (int32_t j = 0; j < 3; ++j) {
				colors[j] = rng.nextVal();
			}

			img.draw_point(x, y, colors);
		}
	}

	cimg_library::CImgDisplay display(img, "Click a point");

	while (!display.is_closed()) {
		display.wait();
	}

	const path dir_path = path("/tmp/generate_2d_sequence");
	const path file_path = dir_path / "temp_img.png";
	img.save(file_path.string().c_str());

	return 0;
}
