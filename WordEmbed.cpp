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
#include <iostream>
#include <fstream>
#include "H5Cpp.h"

#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

typedef unsigned long long u64;

enum OpMode {
	WE_FROM_ONE_HOT,
	WE_FROM_GLOVE,
	WE_SHOW_RESULT,
};

OpMode itMode = WE_FROM_GLOVE;


const int cAsciiMax = '~';
const int cAsciiMin = ' ';
const int cAsciiRange = cAsciiMax - cAsciiMin;

class WordEmbed {
public:
	WordEmbed() {bInit_ = false; }
		  
	void Init(	const string& model_file,
				const string& trained_file,
				const string& word_file_name,
				const string& input_layer_name,
				const string& output_layer_name,
				const string& word_vector_file_name);

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
	int input_layer_dim_;
	vector<int> output_layer_idx_arr_;
	int output_layer_idx_;
	std::vector<float> output_data_;
	int input_layer_bottom_idx_; // currently the index of the array of bottom blobs of the input layer
	int input_layer_delta_idx; // index of array holding the flush/continue flag
	int output_layer_top_idx_; // currently the index of the array of top blobs of the output layer
	int num_data_items_per_;
	int center_of_ngram_;
	int input_layer_top_idx_;
	vector<string> words_;
	vector<vector<float> > words_vecs_;
	string input_layer_name_ ;
	string output_layer_name_;
	string word_vector_file_name_;
	int words_per_input_;
	
	int GetClosestWordIndex(vector<float>& VecOfWord);

};

void WordEmbed::Init(	const string& model_file,
						const string& trained_file,
						const string& word_file_name,
						const string& input_layer_name,
						const string& output_layer_name,
						const string& word_vector_file_name) {
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

	input_layer_name_ = input_layer_name;
	output_layer_name_ = output_layer_name;
	word_vector_file_name_ = word_vector_file_name;
	//output_layer_idx_arr_ = vector<int>(5, -1);

	input_layer_bottom_idx_ = 2;
	center_of_ngram_ = 2;
	input_layer_delta_idx = 1;
	output_layer_top_idx_ = 0;
	words_per_input_ = 1;
	if (itMode == WE_SHOW_RESULT) {
		words_per_input_ = 4;
	}
	
	

	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	
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
	
	if (itMode == WE_FROM_ONE_HOT) {
		input_layer_top_idx_ = center_of_ngram_;
	}
	else {
		input_layer_top_idx_ = 0;
	}

	Blob<float>* input_layer = net_->top_vecs()[input_layer_idx_][input_layer_top_idx_];
	input_layer_dim_ = input_layer->shape(1);

	if (itMode == WE_FROM_ONE_HOT) {
		for (int iln = 0; iln < 5; iln++) {

			char name_ext[] = {'a' + iln, '\0'};
			string output_layer_name = output_layer_name_ + string( (const char *)name_ext);
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
			output_layer_idx_arr_.push_back(output_layer_idx);
		}
	}
	else if (itMode == WE_FROM_GLOVE || itMode == WE_SHOW_RESULT) {
			string output_layer_name = output_layer_name_ ;
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
			int num_input_vals = input_layer_dim_ / words_per_input_;
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

int  WordEmbed::GetClosestWordIndex(vector<float>& VecOfWord)
{
	int num_input_vals = input_layer_dim_ / words_per_input_;
	float MinDiff = num_input_vals * 2.0f;
	int iMinDiff = -1;
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
	}
	return iMinDiff;
}

/* Return the values in the output layer */
bool WordEmbed::Classify() {
	CHECK(bInit_) << "WordEmbed: Init must be called first\n";
	
	Blob<float>* input_layer = net_->top_vecs()[input_layer_idx_][input_layer_top_idx_];
	
	vector<pair<string, vector<float> > > VecArr;
	
	if (itMode == WE_FROM_ONE_HOT) {
		for (int iin = 0; iin < input_layer->shape(1); iin++) {
			net_->ForwardFromTo(0, input_layer_idx_);
			float* p_in = input_layer->mutable_cpu_data();  // ->cpu_data();

			string w = words_[iin];
			std::cerr << w << ",";

			for (int ii = 0; ii < input_layer->shape(1); ii++) {
				*p_in++ = ((ii == iin) ? 1.0f : 0.0f);
			}

			float loss = net_->ForwardFromTo(input_layer_idx_+1, output_layer_idx_arr_[center_of_ngram_]);

			Blob<float>* output_layer = net_->top_vecs()
				[output_layer_idx_arr_[center_of_ngram_]][output_layer_top_idx_]; 
			const float* p_out = output_layer->cpu_data();  
			vector<float> output;
			for (int io = 0; io < output_layer->shape(1); io++) {
				float data = *p_out++;
				std::cerr << data << ",";
				output.push_back(data);
			}
			std::cerr << std::endl;
			VecArr.push_back(make_pair(w, output));
		}
	}
	else if (itMode == WE_SHOW_RESULT) {
		for (int ir = 0; ir < 5000; ir++) {
			net_->ForwardFromTo(0, input_layer_idx_);
			const float* p_in = input_layer->cpu_data();  // ->cpu_data();

			//string w = words_[isym];
			//std::cerr << w << ",";
			const int cNumInputWords = 4;
			const int cNumValsPerWord = 100;
			for (int iw = 0; iw < cNumInputWords; iw++) {
				vector<float> VecOfWord;
				for (int iact = 0; iact < cNumValsPerWord; iact++) {
					//*p_in++ = words_vecs_[isym][iact];
					VecOfWord.push_back(*p_in++);
				}
				int iMinDiff = GetClosestWordIndex(VecOfWord);
				if (iMinDiff != -1) {
					string w = words_[iMinDiff];
					std::cerr << w << " ";
					if (iw == 1) {
						std::cerr << "XXX ";
					}
				}
			}
			std::cerr << std::endl;
			
			float loss = net_->ForwardFromTo(input_layer_idx_+1, output_layer_idx_);

			Blob<float>* output_layer = net_->top_vecs()
				[output_layer_idx_][output_layer_top_idx_]; 
			const float* p_out = output_layer->cpu_data();  
			vector<float> output;
			vector<float> VecOfWord;
			for (int io = 0; io < output_layer->shape(1); io++) {
				float data = *p_out++;
				VecOfWord.push_back(data);
			}
			int iMinDiff = GetClosestWordIndex(VecOfWord);
			if (iMinDiff != -1) {
				string w = words_[iMinDiff];
				std::cerr << "--> " << w << " ";
			}
			std::cerr << std::endl;
			//VecArr.push_back(make_pair(w, output));
		}
	}
	else if (itMode == WE_FROM_GLOVE) {
		for (int isym = 0; isym < words_.size(); isym++) {
			net_->ForwardFromTo(0, input_layer_idx_);
			float* p_in = input_layer->mutable_cpu_data();  // ->cpu_data();

			string w = words_[isym];
			std::cerr << w << ",";
			for (int iact = 0; iact < input_layer_dim_; iact++) {
				*p_in++ = words_vecs_[isym][iact];
			}
			float loss = net_->ForwardFromTo(input_layer_idx_+1, output_layer_idx_);

			Blob<float>* output_layer = net_->top_vecs()
				[output_layer_idx_][output_layer_top_idx_]; 
			const float* p_out = output_layer->cpu_data();  
			vector<float> output;
			for (int io = 0; io < output_layer->shape(1); io++) {
				float data = *p_out++;
				std::cerr << data << ",";
				output.push_back(data);
			}
			std::cerr << std::endl;
			VecArr.push_back(make_pair(w, output));
		}
	}
	
	std::ofstream str_vecs(word_vector_file_name_.c_str());
	if (str_vecs.is_open()) { 
		//str_vecs << VecArr[0].second.size() << " ";
		for (int iv = 0; iv < VecArr.size(); iv++) {
			pair<string, vector<float> >& rec = VecArr[iv];
			str_vecs << rec.first << " ";
			vector<float>& vals = rec.second;
			for (int ir = 0; ir < vals.size(); ir++) {
				str_vecs << vals[ir];
				if (ir == vals.size() - 1) {
					str_vecs << std::endl;
				}
				else {
					str_vecs << " ";
				}
			}
		}
	}



	return true;
}

//std::vector<pair<float, int> > WordEmbed::Predict() {
//
//
//
//
//	return output;
//}

/*
 /home/abba/caffe/toys/ValidClicks/train.prototxt /guten/data/ValidClicks/data/v.caffemodel
 /home/abba/caffe/toys/SimpleMoves/Forward/train.prototxt /devlink/caffe/data/SimpleMoves/Forward/models
 */

#ifdef CAFFE_WORD_EMBED_MAIN
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
	string model_file   = "/home/abba/caffe-recurrent/toys/WordEmbed/VecPredict/train.prototxt";
	string trained_file = "/devlink/caffe/data/WordEmbed/VecPredict/models/v_iter_500000.caffemodel";
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

	itMode = WE_SHOW_RESULT;
	WordEmbed classifier;
	classifier.Init(model_file, trained_file, word_file_name, 
					input_layer_name, output_layer_name,
					word_vector_file_name);
	classifier.Classify();
	
}
#endif // CAFFE_WORD_EMBED_MAIN