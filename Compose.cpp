#include <cstdlib>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "H5Cpp.h"

#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

typedef unsigned long long u64;

const int cAsciiMax = '~';
const int cAsciiMin = ' ';
const int cAsciiRange = cAsciiMax - cAsciiMin;

u64 time_now() {
	struct timeval tv;
	gettimeofday(&tv, NULL);

	u64 ret = tv.tv_usec;
	/* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
	ret /= 1000;

	/* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
	ret += (tv.tv_sec * 1000);
	
	return ret;

}
class Compose {
public:
	Compose() {bInit_ = false; }
		  
	void Init(	const string& model_file,
				const string& trained_file,
				const string& input_layer_name,
				const string& output_layer_name);

	bool  Classify();

private:

	std::vector<pair<float, int> > Predict();


	void Preprocess(const cv::Mat& img,
					std::vector<cv::Mat>* input_channels);

private:
	bool bInit_;
	shared_ptr<Net<float> > net_;
	vector<vector<float> > data_recs_;
	vector<float> label_recs_;
	int input_layer_idx_;
	int output_layer_idx_;
	std::vector<float> output_data_;
	int input_layer_bottom_idx_; // currently the index of the array of bottom blobs of the input layer
	int input_layer_delta_idx; // index of array holding the flush/continue flag
	int output_layer_top_idx_; // currently the index of the array of top blobs of the output layer
	int input_val_next_;
	int num_timestamps_;
	int num_streams_;
	int num_data_items_per_;
	vector<vector<vector<float> > > input_data_;
	vector<vector<float> > put_back_data_; // no suport for streams for now
	bool b_endrec_; // set if a . is seen
	bool b_flush_; // set if last iter had b_endrec_ true

//	cv::Size input_geometry_;
//	int num_in_batch_;
//	int num_channels_;
//	int input_num_channels_;
//	int input_width_;
//	int input_height_;
//	cv::Mat mean_;
//	std::vector<string> labels_;
};

void Compose::Init(	const string& model_file,
					const string& trained_file,
					const string& input_layer_name,
					const string& output_layer_name
				) {
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	
	int input_layer_idx = -1;
	for (size_t layer_id = 0; layer_id < net_->layer_names().size(); ++layer_id) {
		if (net_->layer_names()[layer_id] == input_layer_name) {
			input_layer_idx = layer_id;
			break;
		}
	}
	if (input_layer_idx == -1) {
		LOG(FATAL) << "Unknown layer name " << input_layer_name;			
	}
	
	input_layer_idx_ = input_layer_idx;

	int output_layer_idx = -1;
	for (size_t layer_id = 0; layer_id < net_->layer_names().size(); ++layer_id) {
		if (net_->layer_names()[layer_id] == output_layer_name) {
			output_layer_idx = layer_id;
			break;
		}
	}
	if (output_layer_idx == -1) {
		LOG(FATAL) << "Unknown layer name " << output_layer_name;			
	}
	
	output_layer_idx_ = output_layer_idx;
	
	input_val_next_ = (int)' ' - cAsciiMin;
	input_layer_bottom_idx_ = 0;
	input_layer_delta_idx = 1;
	output_layer_top_idx_ = 0;

	Blob<float>*  input_bottom_vec = net_->top_vecs()[input_layer_idx][input_layer_bottom_idx_];
	num_timestamps_ = input_bottom_vec->shape(0); // parameterize if this might be differemt
	num_streams_ = input_bottom_vec->shape(1);
	num_data_items_per_ = input_bottom_vec->shape(2); 
	input_data_ = vector<vector<vector<float> > >(
			num_timestamps_, 
			vector<vector<float> >(	num_streams_, 
									vector<float>(num_data_items_per_, 0.0f)) );
	for (int it = 0; it < num_timestamps_; it++) {
		for (int is = 0; is < num_streams_; is++) {
			int iact = rand() % num_data_items_per_;
				input_data_[it][is][iact] = 1.0;
		}
	}
	int output_which_hot = rand() % num_data_items_per_;
	vector<float> last_output(num_data_items_per_, 0.0f);
	last_output[output_which_hot] = 1.0f;
	put_back_data_.push_back(last_output);

	
	b_endrec_ = false;
	b_flush_ = true;
	
	/*
	input_layer_bottom_idx_ = input_layer_bottom_idx;
	input_num_channels_ = input_num_channels_idx;
	input_height_ = input_height_idx;
	input_width_ = input_width_idx;
	vector<Blob<float>*>  input_bottom_vec = net_->bottom_vecs()[input_layer_idx];
	num_channels_ = input_bottom_vec[input_layer_bottom_idx_]->shape(input_num_channels_);
	input_geometry_ = cv::Size(	input_bottom_vec[input_layer_bottom_idx_]->shape(input_height_), 
								input_bottom_vec[input_layer_bottom_idx_]->shape(input_width_));
	num_in_batch_ = input_bottom_vec[input_layer_bottom_idx_]->shape(0);
	
	int output_layer_idx = -1;
	for (size_t layer_id = 0; layer_id < net_->layer_names().size(); ++layer_id) {
		if (net_->layer_names()[layer_id] == output_layer_name) {
			output_layer_idx = layer_id;
			break;
		}
	}
	if (output_layer_idx == -1) {
		LOG(FATAL) << "Unknown layer name " << output_layer_name;			
	}
	output_layer_idx_ = output_layer_idx;
	output_layer_top_idx_ = output_layer_top_idx;
	 */

	bInit_ = true;

}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the values in the output layer */
bool Compose::Classify() {
	CHECK(bInit_) << "Compose: Init must be called first\n";
//	input_layer_idx_ = 0;
//	output_layer_idx_ = net_->layer_names().size();

	//std::fill(input_data_.begin(), input_data_.end(), 0.0f);
//	input_data_[input_val_next_] = 1.0f;
	
	std::vector<pair<float, int> > output = Predict();

//	for (uint ir = 0; ir < output.size(); ir++) {
//		RetData.add_data_val(output[ir]);
//	}
//	RetData.set_num_params((int)output.size());
		//std::cout << ReqData.data_val(id) << ", ";
	return true;
}

std::vector<pair<float, int> > Compose::Predict() {

//	int id = 0;
//	for (int ib = 0; ib < num_in_batch_; ib++) {
//		if (id >= data_input.size()) id = 0; // keep the data a multiple of the num in batch
//		for (int ic = 0; ic < num_channels_; ic++) {
//			for (int ih = 0; ih < input_geometry_.height; ih++) {
//				for (int iw = 0; iw < input_geometry_.width; iw++, id++) {
//					*p_begin++ = data_input[id];
//				}
//			}
//		}
//	}
  


	net_->ForwardFromTo(0, input_layer_idx_);
#define COMPOSE_INPUT
#ifdef COMPOSE_INPUT	
	int data_start_pos = num_timestamps_ - put_back_data_.size();
	CHECK_EQ(num_streams_, 1) << "Multiple streams not currently supported\n";
	float* p_deltaf = net_->top_vecs()[input_layer_idx_][input_layer_delta_idx]->mutable_cpu_data(); 
	//p_deltaf[data_start_pos] = 0.0;
	//*p_deltaf = b_flush_ ? 0.0 : 1.0f;
	Blob<float>* input_layer = net_->top_vecs()[input_layer_idx_][input_layer_bottom_idx_];
	float* p_in = input_layer->mutable_cpu_data();  // ->cpu_data();
	float * pp = p_in + (data_start_pos * num_data_items_per_ * num_streams_);
	for (int it = 0; it < put_back_data_.size(); it++) {
		//for (int is = 0; is < num_streams_; is++) { // may be put back once streams are supported
			for (int ii = 0; ii < num_data_items_per_; ii++) {
				//*pp++ = input_data_[it][is][ii];
				*pp++ = put_back_data_[it][ii];
			}
		//}
		if (it == 0) {
			// note we always flush at the start of stack in compose even if the stack
			// size is the same as the number of timestamps because the stack would
			// be larger than the number of timestamps.
			p_deltaf[data_start_pos] = 0.0f;
		}
		else {
			p_deltaf[data_start_pos + it] = 1.0f;
		}
	}
	
#endif // #ifdef COMPOSE_INPUT	
	u64 t1 = time_now();
	float loss = net_->ForwardFromTo(input_layer_idx_+1, net_->layers().size()-1);
	const float* pacc = (net_->top_vecs()[net_->layers().size()-1][0])->cpu_data();
	u64 t2 = time_now() - t1;
#ifndef COMPOSE_INPUT	
	std::cout <<  "\nacc: " << *pacc << std::endl;
#endif // #ifndef COMPOSE_INPUT	

	/* Copy the output layer to a std::vector */
#ifndef COMPOSE_INPUT	
	float* p_delta = (float *)net_->top_vecs()[0][1]->cpu_data();  // ->cpu_data();
	float* p_label = (float *)net_->top_vecs()[0][2]->cpu_data();  // ->cpu_data();
	p_delta++;
	for (int it = 0; it < num_timestamps_; it++) {
		for (int is = 0; is < num_streams_; is++) {
			std::cerr << (char)((int)(*p_delta++) + '0');
		}
	}
	std::cerr << std::endl;
	for (int it = 0; it < num_timestamps_; it++) {
		for (int is = 0; is < num_streams_; is++) {
			std::cerr << (char)((int)(*p_label++) + cAsciiMin);
		}
	}
	std::cerr << std::endl;
//#else
#elif 0
	float* p_delta = (float *)net_->top_vecs()[0][1]->cpu_data();  // ->cpu_data();
	float* p_input = (float *)net_->top_vecs()[0][0]->cpu_data();  // ->cpu_data();
	for (int is = 0; is < num_streams_; is++) {
		std::cerr << '(' << (char)((int)(*p_delta++) + '0') ;
	}
	for (int is = 0; is < num_streams_; is++) {
		vector<pair<float, int> > input;
		for (int ii = 0; ii < num_data_items_per_; ii++) {
			float act = *p_input++;
			input.push_back(make_pair(act, ii));
		}
		std::sort(input.begin(), input.end());
		std::cerr << (char)(input.back().second + cAsciiMin) << ')';
	}
#endif // #ifndef COMPOSE_INPUT	
	Blob<float>* output_layer = net_->top_vecs()[output_layer_idx_][output_layer_top_idx_]; // net_->output_blobs()[0];
	const float* p_out = output_layer->cpu_data();  // ->cpu_data();
	const float * cpp = p_out;
	vector<pair<float, int> > output;
	int output_which_hot = 0;
	for (int it = 0; it < num_timestamps_; it++) {
		for (int is = 0; is < num_streams_; is++) {
			output.clear();
			for (int ii = 0; ii < num_data_items_per_; ii++) {
				float act = *cpp++;
#ifdef COMPOSE_INPUT	
				if (b_endrec_) {
					act *= 1.0f + ((float)(rand() % 4000) / 1000.0f);
				}
				else {
					act *= 1.0f + ((float)(rand() % 50) / 1000.0f);
				}
#endif // #ifndef COMPOSE_INPUT	
				//input_data_[it][is][ii] = 0.0f;
				output.push_back(make_pair(act, ii));
			}
			std::sort(output.begin(), output.end());
#ifdef COMPOSE_INPUT	
			if ((it == (num_timestamps_ - 1)) && (is == (num_streams_ - 1))) {
				output_which_hot = output.back().second;
			}
#endif // #ifdef COMPOSE_INPUT	
#ifndef COMPOSE_INPUT	
			std::cerr << (char)(output.back().second + cAsciiMin);
#endif // #ifndef COMPOSE_INPUT	
		}
	}
#ifdef COMPOSE_INPUT	
	
	std::cerr << (char)(output_which_hot + cAsciiMin);
	vector<float> last_output(num_data_items_per_, 0.0f);
	last_output[output_which_hot] = 1.0f;
	b_flush_ = false;
	if (b_endrec_) {
		b_flush_ = true;
	}
	b_endrec_ = false;
	if (output_which_hot == '.' - cAsciiMin) {
		b_endrec_ = true;
	}
	if (b_flush_) {
		put_back_data_.clear();
	}
	else if (put_back_data_.size() >= num_timestamps_) {
		put_back_data_.erase(put_back_data_.begin());
	}
	put_back_data_.push_back(last_output);
#endif // #ifdef COMPOSE_INPUT	
	
#ifndef COMPOSE_INPUT	
	std::cerr << std::endl;
#endif // #ifndef COMPOSE_INPUT	
//	const float* begin = output_layer->cpu_data();
//	const float* end = begin + output_layer->count();
//	return std::vector<float>(begin, end);
	return output;
}

/*
 /home/abba/caffe/toys/ValidClicks/train.prototxt /guten/data/ValidClicks/data/v.caffemodel
 /home/abba/caffe/toys/SimpleMoves/Forward/train.prototxt /devlink/caffe/data/SimpleMoves/Forward/models
 */

#ifdef CAFFE_COMPOSE_MAIN
int main(int argc, char** argv) {
//	if (argc != 3) {
//		std::cerr << "Usage: " << argv[0]
//				  << " deploy.prototxt network.caffemodel" << std::endl;
//		return 1;
//	 }

	FLAGS_log_dir = "/devlink/caffe/log";
	::google::InitGoogleLogging(argv[0]);
  
//	string model_file   = argv[1];
//	string trained_file = argv[2];
	string model_file   = "/home/abba/caffe-recurrent/toys/LSTMTrain/words/train.prototxt";
	string trained_file = "/devlink/caffe/data/LSTMTrain/words/models/w_iter_10000.caffemodel";
	
	string input_layer_name = "data";
	string output_layer_name = "predict";
	int input_data_idx = 0;
	int input_label_idx = 1;

	Compose classifier;
	classifier.Init(model_file, trained_file, input_layer_name, 
					output_layer_name);
	for (int i = 0; i < 1000; i++) {
		classifier.Classify();
	}
	
}
#endif // CAFFE_COMPOSE_MAIN