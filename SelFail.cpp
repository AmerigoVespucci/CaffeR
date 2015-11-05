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



class SelFail {
public:
	SelFail() {bInit_ = false; }
		  
	void Init(	const string& model_file,
				const string& trained_file,
				const string& input_layer_name,
					const string& out_file_ext,
					const int num_hd5_recs);

	bool  Classify();

private:
//	void SetMean(const string& mean_file);

	std::vector<float> Predict(const vector<float>& data_input);

//	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
					std::vector<cv::Mat>* input_channels);

private:
	bool bInit_;
	shared_ptr<Net<float> > net_;
	vector<vector<float> > data_recs_;
	vector<float> label_recs_;
	int input_layer_idx_;
	int output_layer_idx_;
	string out_file_name_;
	int num_hd5_recs_;

//	cv::Size input_geometry_;
//	int num_in_batch_;
//	int num_channels_;
//	int input_layer_bottom_idx_; // currently the index of the array of bottom blobs of the input layer
//	int output_layer_top_idx_; // currently the index of the array of top blobs of the output layer
//	int input_num_channels_;
//	int input_width_;
//	int input_height_;
//	cv::Mat mean_;
//	std::vector<string> labels_;
//	std::vector<float> input_data_;
};

void SelFail::Init(	const string& model_file,
					const string& trained_file,
					const string& input_layer_name,
					const string& out_file_ext,
					const int num_hd5_recs
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
	out_file_name_ = out_file_ext;
	num_hd5_recs_ = num_hd5_recs;
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
bool SelFail::Classify() {
	CHECK(bInit_) << "SelFail: Init must be called first\n";
	input_layer_idx_ = 0;
	output_layer_idx_ = net_->layer_names().size();
	
	int num_bad = 0;
	vector<Blob<float>*> bottom_vec;
//	shared_ptr<Layer<float> > l = (net_->layers()[0]);
//	shared_ptr<HDF5DataLayer<float> > pHdfLayer = boost::static_pointer_cast<HDF5DataLayer<float> >(l);
//	int NumDataItems = pHdfLayer->ExactNumBottomBlobs();
	for (int i = 0; i < num_hd5_recs_; i++) {
		float Score;
		float loss = 0;
		float iter_loss;
		const vector<Blob<float>*>& result = net_->Forward(bottom_vec, &iter_loss);
		for (int j = 0; j < result.size(); ++j) {
			const float* result_vec = result[j]->cpu_data();
			if (net_->blob_names()[net_->output_blob_indices()[j]] == "accuracy" ) {
				bool b_add_rec = false;
				bool this_bad = false;
				Score = result_vec[0];
				if (Score < 0.9f) {
					b_add_rec = true;
					num_bad++;
					this_bad = true;
					std::cerr << "Record # " << i << " bad!\n";
				}
//				else {
//					if ((rand() % 98) == 0) {
//						b_add_rec = true;
//					}
//				}
				
				if (b_add_rec) {
					//FailedRecs.push_back(i);
					data_recs_.push_back(vector<float>());
					vector<float>& rec = data_recs_.back();
					Blob<float>* input_layer_blob = net_->top_vecs()[input_layer_idx_][0]; // net_->output_blobs()[0];
					const float* begin = input_layer_blob->cpu_data();
					int num_items = input_layer_blob->count();
					for (int in = 0; in < num_items; in++) {
						rec.push_back(*(begin + in));
					}
					Blob<float>* label_blob = net_->top_vecs()[input_layer_idx_][1]; // net_->output_blobs()[0];
					const float* plabel = label_blob->cpu_data();
					label_recs_.push_back(*plabel);
					
				}
			}
//			for (int k = 0; k < result[j]->count(); ++k) {
//				//test_score.push_back(result_vec[k]);
//				//test_score_output_id.push_back(j);
//			}
		}
	
//    if (i == 0) {
//		for (int j = 0; j < result.size(); ++j) {
//			const float* result_vec = result[j]->cpu_data();
//			for (int k = 0; k < result[j]->count(); ++k) {
//				test_score.push_back(result_vec[k]);
//				//test_score_output_id.push_back(j);
//			}
//		}
//    } else {
//		int idx = 0;
//		for (int j = 0; j < result.size(); ++j) {
//			const float* result_vec = result[j]->cpu_data();
//			for (int k = 0; k < result[j]->count(); ++k) {
//				test_score[idx++] += result_vec[k];
//			}
//		}
//    }
	}
	std::cerr << "Found " << num_bad << " bad recs\n";

	int num_labels_per_rec = 1;
	int NumPieceTypes = 2;
	//int num_items_per_rec_pos = 16;
	int num_items_per_rec_pos = 4;
	int num_records = data_recs_.size();
	int x_board_size = 8;
	int y_board_size = 8;

	const int cDataRank =  4;
	const H5std_string	DATASET_NAME("data");
	const H5std_string	LABELSET_NAME("label");
	hsize_t dims[cDataRank];               // dataset dimensions
	dims[3] = x_board_size;
	dims[2] = y_board_size;
	dims[1] = num_items_per_rec_pos;
	dims[0] = num_records;
	const int cLabelRank = 2;
	hsize_t label_dims[cLabelRank];               // dataset dimensions
	label_dims[1] = num_labels_per_rec;
	label_dims[0] = num_records;

	float * pDataSet = new float[num_records * num_items_per_rec_pos * x_board_size * y_board_size];
	float * ppd = pDataSet;
	float * plabels = new float[num_records * num_labels_per_rec];
	float * ppl = plabels;

	for (int ir=0; ir < num_records; ir++) {
		int rpos = 0;
		vector<float> & rec = data_recs_[ir];
		for (uint ipt = 0; ipt < num_items_per_rec_pos; ipt++) {
			for (int ypos = 0; ypos < y_board_size; ypos++) {
				for (int xpos = 0; xpos < x_board_size; xpos++, rpos++) {
					*(ppd++) = rec[rpos];
				}
			}
		}
		*(ppl++) = label_recs_[ir];
	}

	H5std_string fname = out_file_name_;
	H5File h5file(fname, H5F_ACC_TRUNC);
	DataSpace dataspace(cDataRank, dims);
	DataSpace labelspace(cLabelRank, label_dims);
	DataSet dataset = h5file.createDataSet(DATASET_NAME, PredType::IEEE_F32LE, dataspace);

	dataset.write(pDataSet, PredType::IEEE_F32LE);

	DataSet labelset = h5file.createDataSet(LABELSET_NAME, PredType::IEEE_F32LE, labelspace);

	labelset.write(plabels, PredType::IEEE_F32LE);
	

	delete pDataSet;
	delete plabels;


	/*
	CHECK_EQ(ReqData.num_params(), ReqData.data_val_size()) << "SelFail data corrupted";
	//std::cerr << ReqData.num_params() << " data items reported. Received: " << ReqData.data_val_size() << ".\n";
	input_data_.clear();
	for (int id = 0; id < ReqData.num_params(); id++) {
		input_data_.push_back(ReqData.data_val(id));

		//std::cout << ReqData.data_val(id) << ", ";
	}
			//std::cout << std::endl;
	std::vector<float> output = Predict(input_data_);

	for (uint ir = 0; ir < output.size(); ir++) {
		RetData.add_data_val(output[ir]);
	}
	RetData.set_num_params((int)output.size());
	*/
	return true;
}

std::vector<float> SelFail::Predict(const vector<float>& data_input) {
	/*
	Blob<float>* input_layer = net_->bottom_vecs()[input_layer_idx_][input_layer_bottom_idx_];
	float* p_begin = input_layer->mutable_cpu_data();  // ->cpu_data();

	int id = 0;
	for (int ib = 0; ib < num_in_batch_; ib++) {
		if (id >= data_input.size()) id = 0; // keep the data a multiple of the num in batch
		for (int ic = 0; ic < num_channels_; ic++) {
			for (int ih = 0; ih < input_geometry_.height; ih++) {
				for (int iw = 0; iw < input_geometry_.width; iw++, id++) {
					*p_begin++ = data_input[id];
				}
			}
		}
	}
	*/
	net_->ForwardFromTo(input_layer_idx_, output_layer_idx_); 

	/* Copy the output layer to a std::vector */
	int output_layer_top_idx = 0;
	Blob<float>* output_layer = net_->top_vecs()[output_layer_idx_][output_layer_top_idx]; // net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	return std::vector<float>(begin, end);
}

/*
 /home/abba/caffe/toys/ValidClicks/train.prototxt /guten/data/ValidClicks/data/v.caffemodel
 /home/abba/caffe/toys/SimpleMoves/Forward/train.prototxt /devlink/caffe/data/SimpleMoves/Forward/models
 */

#ifdef CAFFE_SEL_FAIL_MAIN
int main(int argc, char** argv) {
	if (argc != 3) {
		std::cerr << "Usage: " << argv[0]
				  << " deploy.prototxt network.caffemodel" << std::endl;
		return 1;
	 }

	FLAGS_log_dir = "/devlink/caffe/log";
	::google::InitGoogleLogging(argv[0]);
  
//	string model_file   = argv[1];
//	string trained_file = argv[2];
	string model_file   = "/home/abba/caffe/toys/SimpleMoves/Forward/train.prototxt";
	string trained_file = "/devlink/caffe/data/SimpleMoves/Forward/models/f_iter_30000.caffemodel";
	
	int NumCasesInH5 = 100000; // 500000;
	//string out_file_ext = "/guten/data/ValidClicks/data/teste.h5";
	string out_file_ext = "/devlink/caffe/data/SimpleMoves/Forward/data/teste.h5";
	string input_layer_name = "data";
	int input_data_idx = 0;
	int input_label_idx = 1;

	SelFail classifier;
	classifier.Init(model_file, trained_file, input_layer_name, 
					out_file_ext, NumCasesInH5);
	classifier.Classify();
	
}
#endif // CAFFE_ONE_RUN_MAIN