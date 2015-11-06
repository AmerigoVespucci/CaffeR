#include <cstdlib>
#include <boost/random.hpp>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <caffe/caffe.hpp>
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include <opencv2/core/core.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include "H5Cpp.h"


#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

typedef unsigned long long u64;

class NGNet {
public:
	NGNet(	const string& model_file,
				const string& trained_file,
				const string& input_layer_name,
				const string& output_layer_name) {
		model_file_ = model_file;
		trained_file_ = trained_file;
		input_layer_name_ = input_layer_name;
		output_layer_name_ = output_layer_name;
		b_self_launched = true;
	}
	
	NGNet(	shared_ptr<NGNet> launcher,
			int mod_param_layer_idx,
			int mod_layer_idx,
			int mod_layer_post_idx) {
		model_file_ = launcher->model_file_;
		input_layer_name_ = launcher->input_layer_name_;
		output_layer_name_ = launcher->output_layer_name_;
		b_self_launched = false;
		mod_param_layer_idx_ = mod_param_layer_idx;
		mod_layer_idx_ = mod_layer_idx;
		mod_layer_post_idx_ = mod_layer_post_idx;
		launcher_ = launcher;
	}

	void Launch();
	void Init();
	Blob<float>* GetVec(bool b_top, int layer_idx, int branch_idx);
	Blob<float>* GetInputVec() {
		return GetVec(true, input_layer_idx_, input_layer_top_idx_);
	}
	Blob<float>* GetOutputVec() {
		return GetVec(true, output_layer_idx_, output_layer_top_idx_);
	}
	void PrepForInput() {
		net_->ForwardFromTo(0, input_layer_idx_);
	}
	float ComputeOutput() {
		return net_->ForwardFromTo(input_layer_idx_+1, output_layer_idx_);
	}
	float ComputeRemainingLayers() {
		return net_->ForwardFromTo(output_layer_idx_+1, net_->layers().size() - 1);
	}
	int input_layer_dim() { return input_layer_dim_; }
	int input_layer_idx() { return input_layer_idx_; }
	
private:
	shared_ptr<Net<float> > net_;
	//shared_ptr<Net<float> > new_net_;
	int input_layer_idx_;
	int input_layer_top_idx_; // currently the index of the array of top_vectors for this net
	int output_layer_idx_;
	int output_layer_top_idx_; // currently the index of the array of top_vectors for this net
	string model_file_;
	string trained_file_;
	string input_layer_name_ ;
	string output_layer_name_;
	int input_layer_dim_;
	bool b_self_launched;
	shared_ptr<NGNet> launcher_; // dangerous pointer. For now, promise to use only in Init and never again
	int mod_param_layer_idx_;
	int mod_layer_idx_;
	int mod_layer_post_idx_;
};

class NetGen {
public:
	NetGen() {bInit_ = false; }

	void PreInit();
	void Init(	vector<shared_ptr<NGNet> >& nets,
				const string& word_file_name,
				const string& word_vector_file_name);

	bool  Classify();

private:

	std::vector<pair<float, int> > Predict();


	void Preprocess(const cv::Mat& img,
					std::vector<cv::Mat>* input_channels);

private:
	bool bInit_;
	vector<shared_ptr<NGNet> > nets_;
	vector<vector<float> > data_recs_;
	vector<float> label_recs_;
	vector<string> words_;
	vector<vector<float> > words_vecs_;
	string word_vector_file_name_;
	int words_per_input_;
	
	int GetClosestWordIndex(vector<float>& VecOfWord, int num_input_vals, 
							vector<pair<float, int> >& SortedBest,
							int NumBestKept);

};

// returns a rnd num from -0.5f to 0.5f uniform dist, don't know (care) about endpoints
float rn(void)
{
  boost::mt19937 rng(43);
  static boost::uniform_01<boost::mt19937> zeroone(rng);
  return ((float)zeroone() - 0.5f);
}

void NGNet::Launch()
{
	if (b_self_launched) {
		net_.reset(new Net<float>(model_file_, TEST));
		net_->CopyTrainedLayersFrom(trained_file_);	
	}
	else {
		NetParameter param;
		CHECK(ReadProtoFromTextFile(model_file_, &param))
			<< "Failed to parse NetParameter file: " << model_file_;
		LayerParameter* layer_param = param.mutable_layer(mod_param_layer_idx_);
		if (layer_param->has_inner_product_param()) {
			InnerProductParameter* inner_product_param = layer_param->mutable_inner_product_param();
			int num_output = inner_product_param->num_output();
			if (num_output > 0) {
				inner_product_param->set_num_output(num_output * 2); 
			}
		}
		param.mutable_state()->set_phase(TEST);
		net_.reset(new Net<float>(param));

		for (int il = 0; il < launcher_->net_->layers().size(); il++) {
			Layer<float>* layer = launcher_->net_->layers()[il].get();
			for (int ib=0; ib < layer->blobs().size(); ib++) {
				Blob<float>* weights = layer->blobs()[ib].get() ;
				Blob<float>* new_weights = net_->layers()[il]->blobs()[ib].get() ;
				if (weights->count() == 0) {
					continue;
				}
				if (weights->count() == new_weights->count()) {
					const float * pw = weights->cpu_data();
					float * pnw = new_weights->mutable_cpu_data();
					for (int iw = 0; iw < weights->count(); iw++) {
						pnw[iw] = pw[iw];
					}
				}
				else if (new_weights->count() == (weights->count() * 2)) {
					float min_val = FLT_MAX;
					float max_val = -FLT_MAX;
					const float * pw = weights->cpu_data();
					float * pnw = new_weights->mutable_cpu_data();
					for (int iw = 0; iw < weights->count(); iw++) {
						float v = pw[iw];
						if (v > max_val) max_val = v;
						if (v < min_val) min_val = v;
					}
					float cFrac = 0.01f;
					float rmod = (max_val - min_val) * cFrac;
					float rtwice = 1.0f;
					if (il == mod_layer_idx_) {
						rtwice = 0.5f;
					}
					int num_inputs = 1;
					if (weights->num_axes() > 1) { 
						num_inputs = weights->shape(1);
					}
					int num_outputs = weights->shape(0);
					for (int jw = 0; jw < num_outputs; jw++) {
						for (int iw = 0; iw < num_inputs; iw++) {
							float adj = rn() * rmod;
							//adj = 0.0f;
							int jwo = jw * 2;
							if (il == mod_layer_idx_) {
								pnw[(jwo * num_inputs) + iw] 
									= (pw[(jw * num_inputs) + iw] * rtwice) + adj;
								pnw[((jwo + 1) * num_inputs) + iw] 
									= (pw[(jw * num_inputs) + iw] * rtwice) - adj;
							}
							else if (il == mod_layer_post_idx_) {
								int ii = (jw * num_inputs) + iw;
								pnw[ii * 2] = pw[ii];
								pnw[(ii * 2) + 1] = pw[ii];
							}
						}
					}
				}
//				if (il == 6) {
//					const float * pw = weights->cpu_data();
//					std::cerr << "old weights: ";
//					for (int iw = 0; iw < weights->count(); iw++) {
//						std::cerr << pw[iw] << ", ";
//					}
//					std::cerr << std::endl;
//					const float * pnw = new_weights->cpu_data();
//					std::cerr << "new weights: ";
//					for (int iw = 0; iw < new_weights->count(); iw++) {
//						std::cerr << pnw[iw] << ", ";
//					}
//					std::cerr << std::endl;
//				}

			}
		}
	}
}

void NGNet::Init(	) {

	Launch();
	input_layer_top_idx_ = 0;
	output_layer_top_idx_ = 0;
	
	/* Load the network. */
	
	int input_layer_idx = -1;
	for (size_t layer_id = 0; layer_id < net_->layer_names().size(); ++layer_id) {
		if (net_->layer_names()[layer_id] == input_layer_name_) {
			input_layer_idx = layer_id;
			break;
		}
	}
	if (input_layer_idx == -1) {
		LOG(FATAL) << "Unknown layer name " << input_layer_name_;			
	}

	input_layer_idx_ = input_layer_idx;
	
	input_layer_top_idx_ = 0;

	Blob<float>* input_layer = net_->top_vecs()[input_layer_idx_][input_layer_top_idx_];
	input_layer_dim_ = input_layer->shape(1);

	int output_layer_idx = -1;
	for (size_t layer_id = 0; layer_id < net_->layer_names().size(); ++layer_id) {
		if (net_->layer_names()[layer_id] == output_layer_name_) {
			output_layer_idx = layer_id;
			break;
		}
	}
	if (output_layer_idx == -1) {
		LOG(FATAL) << "Unknown layer name " << output_layer_name_;			
	}
	output_layer_idx_ = output_layer_idx;
	
	
}

Blob<float>* NGNet::GetVec(bool b_top, int layer_idx, int branch_idx)
{
	if (b_top) {
		return net_->top_vecs()[layer_idx][branch_idx];
	}
	else {
		return net_->bottom_vecs()[layer_idx][branch_idx];
	}
}


void NetGen::PreInit()
{
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif
}

void NetGen::Init(	vector<shared_ptr<NGNet> >& nets,
						const string& word_file_name,
						const string& word_vector_file_name) {

	word_vector_file_name_ = word_vector_file_name;
	//output_layer_idx_arr_ = vector<int>(5, -1);

	words_per_input_ = 1;
	words_per_input_ = 4;
	
	nets_ = nets;
	
	for (int in = 0; in < nets.size(); in++) {
		shared_ptr<NGNet> net = nets[in];
		net->Init();
	}

	
	
	std::ifstream str_words(word_file_name.c_str(), std::ifstream::in);
	if (str_words.is_open() ) {
		string ln;
		//for (int ic = 0; ic < cVocabLimit; ic++) {
		while (str_words.good()) {
			string w;
			getline(str_words, ln, ' ');
			//VecFile >> w;
			w = ln;
			if (w.size() == 0) {
				break;
			}
			words_.push_back(w);
			words_vecs_.push_back(vector<float>());
			vector<float>& curr_vec = words_vecs_.back();
			int num_input_vals = nets[0]->input_layer_dim() / words_per_input_;
			for (int iwv = 0; iwv < num_input_vals; iwv++) {
				if (iwv == num_input_vals - 1) {
					getline(str_words, ln);
				}
				else {
					getline(str_words, ln, ' ');
				}
				float wv;
				//wv = stof(ln);
				wv = (float)atof(ln.c_str());
				curr_vec.push_back(wv);
			}

		}
	}
	//Blob<float>*  input_bottom_vec = net_->top_vecs()[input_layer_idx][input_layer_bottom_idx_];
	

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

int  NetGen::GetClosestWordIndex(	vector<float>& VecOfWord, int num_input_vals, 
									vector<pair<float, int> >& SortedBest, int NumBestKept)
{
	float MinDiff = num_input_vals * 2.0f;
	int iMinDiff = -1;
	float ThreshDiff = MinDiff;
	for (int iwv =0; iwv < words_vecs_.size(); iwv++ ) {
		float SumDiff = 0.0f;
		for (int iv = 0; iv < num_input_vals; iv++) {
			float Diff = VecOfWord[iv] - words_vecs_[iwv][iv];
			SumDiff += Diff * Diff;
		}
		if (SumDiff < MinDiff) {
			MinDiff = SumDiff;
			iMinDiff = iwv;
		}
		if (SumDiff < ThreshDiff) {
			SortedBest.push_back(make_pair(SumDiff, iwv));
			std::sort(SortedBest.begin(), SortedBest.end());
			if (SortedBest.size() > NumBestKept) {
				SortedBest.pop_back();
				ThreshDiff = SortedBest.back().first;
			}
		}
	}
	return iMinDiff;
}

/* Return the values in the output layer */
bool NetGen::Classify() {
	CHECK(bInit_) << "NetGen: Init must be called first\n";
	
	
	int CountMatch = 0;
	int NumTestRecs = 1000;
	
	for (int in = 0; in < 2; in++) {
		float loss_sum = 0.0f;

		for (int ir = 0; ir < NumTestRecs; ir++) {
			nets_[in]->PrepForInput();

			float loss = nets_[in]->ComputeOutput();
			loss = nets_[in]->ComputeRemainingLayers();
			Blob<float>* LastLayer = nets_[in]->GetVec(true, 10, 0);
			float acc_loss = *(LastLayer->cpu_data());

			loss_sum += acc_loss;
			//float loss = net_->ForwardFromTo(input_layer_idx_+1, output_layer_idx_);

		}

		std::cerr << "Avergage loss: " << loss_sum / NumTestRecs << std::endl;
	}

	return true;
}


/*
 /home/abba/caffe/toys/ValidClicks/train.prototxt /guten/data/ValidClicks/data/v.caffemodel
 /home/abba/caffe/toys/SimpleMoves/Forward/train.prototxt /devlink/caffe/data/SimpleMoves/Forward/models
 */

#ifdef CAFFE_NET_GEN_MAIN
int main(int argc, char** argv) {
//	if (argc != 3) {
//		std::cerr << "Usage: " << argv[0]
//				  << " deploy.prototxt network.caffemodel" << std::endl;
//		return 1;
//	 }

	FLAGS_log_dir = "/devlink/caffe/log";
	::google::InitGoogleLogging(argv[0]);
  
//	string model_file   = "/home/abba/caffe-recurrent/toys/NetGen/VecPredict/train.prototxt";
//	string trained_file = "/devlink/caffe/data/NetGen/VecPredict/models/v_iter_500000.caffemodel";
	string word_file_name = "/devlink/caffe/data/WordEmbed/VecPredict/data/WordList.txt";
	string word_vector_file_name = "/devlink/caffe/data/WordEmbed/VecPredict/data/WordVectors.txt";
//	string model_file   = "/home/abba/caffe-recurrent/toys/LSTMTrain/WordToPos/train.prototxt";
//	string trained_file = "/devlink/caffe/data/LSTMTrain/WordToPos/models/a_iter_1000000.caffemodel";
//	string word_file_name = "/devlink/caffe/data/LSTMTrain/WordToPos/data/WordList.txt";
//	string word_vector_file_name = "/devlink/caffe/data/LSTMTrain/WordToPos/data/WordVectors.txt";
	string input_layer_name = "data";
	string output_layer_name = "SquashOutput";
	
	int input_data_idx = 0;
	int input_label_idx = 1;

	const int mod_param_layer_idx = 2; // 5; // 2
	const int mod_layer_idx = 2; // 4; // 2
	const int mod_layer_idx_post = 4; // 6; // 4

	NetGen classifier;
	classifier.PreInit();
	vector<shared_ptr<NGNet> > nets;
	nets.push_back(shared_ptr<NGNet>(new NGNet(
		"/devlink/github/test/toys/NetGen/GramPosValid/train.prototxt",
		"/devlink/caffe/data/NetGen/GramPosValid/models/g_iter_120921.caffemodel",
		"data", "SquashOutput")));
	nets.push_back(shared_ptr<NGNet>(new NGNet(	nets[0], mod_param_layer_idx, 
												mod_layer_idx, mod_layer_idx_post)));

//	nets.push_back(NGNet(
//		"/home/abba/caffe-recurrent/toys/WordEmbed/GramValid/train.prototxt",
//		"/devlink/caffe/data/WordEmbed/GramValid/models/g_iter_500000.caffemodel",
//		"data", "SquashOutput"));
	classifier.Init(nets,
					word_file_name, 
					word_vector_file_name);
	classifier.Classify();
	
}
#endif // CAFFE_MULTINET_MAIN