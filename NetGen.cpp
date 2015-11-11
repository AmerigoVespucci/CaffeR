#include <cstdlib>
#include <boost/random.hpp>
#include <boost/timer/timer.hpp>
#include <boost/chrono/duration.hpp>
#include <boost/algorithm/string.hpp>
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
typedef boost::chrono::duration<double> sec;

SolverAction::Enum MakeSolverStop()  {
  return SolverAction::STOP;
}

SolverAction::Enum SolverKeepGoing()  {
  return SolverAction::NONE;
}


class NetGenSolver : public SGDSolver<float> {
public:
	explicit NetGenSolver(const SolverParameter& param)
		: SGDSolver<float>(param) { Init(); }
	explicit NetGenSolver(const string& param_file)
		: SGDSolver<float>(param_file) { Init(); }
	virtual ~NetGenSolver() {
		std::cerr << "NetGenSolver deleted\n";
	}
	
	void set_loss_layer(int loss_layer) { loss_layer_ = loss_layer; }
	void set_run_time(double run_time) {run_time_ = run_time; }
	float get_avg_loss() { return loss_sum_ / loss_sum_count_; }
	void get_net(shared_ptr<Net<float> >& net) {net = this->net_; }
	Net<float> * get_test_net() {return this->test_nets_[0].get(); }
	void reset_loss_sum() {	SetActionFunction(SolverKeepGoing);
							timer_.stop(); timer_.start();
							loss_sum_= 0.0f; loss_sum_count_ = 0.0f; 
							iter_ = 0; }

protected:
	int loss_layer_;
	float loss_sum_;
	float loss_sum_count_;
	double run_time_;
	boost::timer::cpu_timer timer_;
	
	
	virtual void ApplyUpdate();
	void Init();
  
  DISABLE_COPY_AND_ASSIGN(NetGenSolver);
};

void NetGenSolver::Init()
{
	loss_layer_ = -1;
	loss_sum_= 0;
	loss_sum_count_ = 0;
	run_time_ = 0.0;
	SetActionFunction(SolverKeepGoing);
}
 
void NetGenSolver::ApplyUpdate()
{
	SGDSolver::ApplyUpdate();
	CHECK_GT(loss_layer_, 0) << "Error: Loss layer for NetGenSolver must be set before calling Solve\n";
	
	float loss = *(net_->top_vecs()[loss_layer_][0]->cpu_data());
	loss_sum_ += loss;
	loss_sum_count_++;
	sec seconds = boost::chrono::nanoseconds(timer_.elapsed().user + timer_.elapsed().system);
	if (seconds.count() > run_time_) {
		//iter_ = INT_MAX - 100;
		requested_early_exit_ = true;
		SetActionFunction(MakeSolverStop);
	}
}

class NGNet {
public:
	NGNet(	) {
//		b_self_launched_ = true;
	}
	
//	NGNet(	shared_ptr<NGNet> launcher
//		) {
//		b_self_launched_ = false;
//		launcher_ = launcher;
//	}
	virtual ~NGNet() {
		std::cerr << "NGNet destructor called\n";
	}

	void Gen(int layer_1_nodes, int layer_2_nodes, int mod_idx_idx);
	float DoRun(bool bIntersection);
	void SetWeights(NGNet* p_ng_net);
	int get_ipx_layer_idx(int idx_idx) { return ip_layer_idx_arr_[idx_idx]; }
	

	
	void Launch();
	void Init();
	
private:
	shared_ptr<NetGenSolver> solver_;
	shared_ptr<Net<float> > net_;
	//shared_ptr<Net<float> > new_net_;
	//bool b_self_launched_;
	//shared_ptr<NGNet> launcher_; // dangerous pointer. For now, promise to use only in Init and never again
	vector<int> ip_layer_idx_arr_;
	int ip_layer_idx_idx_;
};

class NetGen {
public:
	NetGen() {bInit_ = false; }

	void PreInit();
	void Init();

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

void NGNet::SetWeights(NGNet* p_ng_net)
{
	Net<float> * launch_net = p_ng_net->net_.get();
	
	int mod_idx = ip_layer_idx_arr_[ip_layer_idx_idx_];
	int mod_post_idx = ip_layer_idx_arr_[ip_layer_idx_idx_+1];
	for (int il = 0; il < launch_net->layers().size(); il++) {
		Layer<float>* layer = launch_net->layers()[il].get();
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
				if (il == mod_idx) {
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
						if (il == mod_idx) {
							pnw[(jwo * num_inputs) + iw] 
								= (pw[(jw * num_inputs) + iw] * rtwice) + adj;
							pnw[((jwo + 1) * num_inputs) + iw] 
								= (pw[(jw * num_inputs) + iw] * rtwice) - adj;
						}
						else if (il == mod_post_idx) {
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

void NGNet::Init(	) {

	
	
}



void NetGen::PreInit()
{
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif
}

const bool cb_ReLU = true;
const bool cb_Sigmoid = true;
const bool cb_drop = true;

const string cHD5Str1 = "name: \"GramPosValid\"\n"
					"layer {\n"
					"	name: \"data\"\n"
					"	type: \"HDF5Data\"\n"
					"	top: \"data\"\n"
					"	top: \"label\"\n"	
					"	include {\n";
const string cHD5Str2 = "	}\n"
					"	hdf5_data_param {\n"
					"		source: ";
const string cHD5StrBatch128 = "\n"
					"		batch_size: 128\n"
					"	}\n"
					"}\n";
const string cHD5StrBatch1 = "\n"
					"		batch_size: 1\n"
					"	}\n"
					"}\n";

string CreateHD5TrainStr(string train_file)
{
	string ret_str = cHD5Str1 
					+ "		phase: TRAIN\n"
					+ cHD5Str2
					+ "\"" + train_file + "\""
					+ cHD5StrBatch128;
	
	return ret_str;
					
}

string CreateHD5TestStr(string test_file)
{
	string ret_str = cHD5Str1 
					+ "		phase: TEST\n"
					+ cHD5Str2
					+ "\"" + test_file + "\""
					+ cHD5StrBatch1;
					
	return ret_str;
}

string CreateReLUStr()
{
	return string(	"layer {\n"
					"  name: \"squash#id#\"\n"
					"  type: \"ReLU\"\n"
					"  bottom: \"ip#id#\"\n"
					"  top: \"ip#id#s\"\n"
					"}\n");

}

string CreateSigmoidStr()
{
	return string(	"layer {\n"
					"  name: \"squash#id#\"\n"
					"  type: \"Sigmoid\"\n"
					"  bottom: \"ip#id#\"\n"
					"  top: \"ip#id#s\"\n"
					"}\n");

}

string CreateDropStr()
{
	return string(	"layer {\n"
					"  name: \"drop#id#\"\n"
					"  type: \"Dropout\"\n"
					"  bottom: \"ip#id#s\"\n"
					"  top: \"ip#id#s\"\n"
					"  dropout_param {\n"
					"    dropout_ratio: #drop_rate#\n"
					"  }\n"
					"  include {\n"
					"    phase: TRAIN\n"
					"  }\n"
					"}\n");


}

string AddSoftmaxAndAccuracyStr (int prev_id, int& layers_so_far )
{
	string modi = 
		"layer {\n"
		"  name: \"loss\"\n"
		"  type: \"SoftmaxWithLoss\"\n"
		"  bottom: \"ip#0#s\"\n"
		"  bottom: \"label\"\n"
		"  top: \"sm-loss\"\n"
		"}\n"
		"layer {\n"
		"  name: \"accuracy\"\n"
		"  type: \"Accuracy\"\n"
		"  bottom: \"ip#0#s\"\n"
		"  bottom: \"label\"\n"
		"  top: \"accuracy\"\n"
		"}\n";

	string sid = boost::lexical_cast<string>(prev_id);
	modi = boost::replace_all_copy(modi, "#0#", sid);
	
	layers_so_far += 2;
	return modi;

}

string AddInnerProductStr(	bool b_ReLU, bool b_Sigmoid, bool b_drop, int id, 
							int num_output, int& layers_so_far, float dropout) 
{
	string frame = 
		"layer {\n"
		"  name: \"ip#id#\"\n"
		"  type: \"InnerProduct\"\n"
		"  bottom: \"#prev_top#\"\n"
		"  top: \"ip#id#\"\n"
		"  param {\n"
		"    lr_mult: 1\n"
		"  }\n"
		"  param {\n"
		"    lr_mult: 2\n"
		"  }\n"
		"  inner_product_param {\n"
		"    num_output: #num_output#\n"
		"    weight_filler {\n"
		"      type: \"xavier\"\n"
		"    }\n"
		"    bias_filler {\n"
		"      type: \"constant\"\n"
		"    }\n"
		"  }\n"
		"}\n";
	string modi =		frame + (b_ReLU ? CreateReLUStr() : "") 
					+	(b_Sigmoid ? CreateSigmoidStr() : "") 
					+	(b_drop ? CreateDropStr() : "");
	
	string sid = boost::lexical_cast<string>(id);
	string s_input = string("ip") + boost::lexical_cast<string>(id-1) + "s";
	if (id == 1) {
		s_input = "data";
	}
	string s_num_output = boost::lexical_cast<string>(num_output);
	string s_dropout = boost::lexical_cast<string>(dropout);
	modi = boost::replace_all_copy(modi, "#id#", sid);
	modi = boost::replace_all_copy(modi, "#num_output#", s_num_output);
	modi = boost::replace_all_copy(modi, "#drop_rate#", s_dropout);
	modi = boost::replace_all_copy(modi, "#prev_top#", s_input);
	
	layers_so_far += 3;
	
	return modi;
	
}

void NGNet::Gen(int layer_1_nodes, int layer_2_nodes, int mod_idx_idx ) {
	
	string input =	
					"test_iter: 10\n"
					"test_interval: 10000\n" // pro forma
					"base_lr: 0.01\n"
					"lr_policy: \"step\"\n"
					"gamma: 0.9\n"
					"stepsize: 100000\n"
					"display: 2000\n"
					"max_iter: 500000\n"
					"momentum: 0.9\n"
					"weight_decay: 0.0005\n"
					"snapshot: 100000\n"
					"snapshot_prefix: \"/devlink/caffe/data/NetGen/GramPosValid/models/g\"\n"
					"solver_mode: CPU\n";
	SolverParameter solver_param;
	bool success = google::protobuf::TextFormat::ParseFromString(input, &solver_param);
	NetParameter* net_param = solver_param.mutable_train_net_param();
	string net_def = CreateHD5TrainStr("/devlink/github/test/toys/NetGen/GramPosValid/train_list.txt");
	int num_layers_so_far = 1;
	const int c_num_data_split_layers = 1;
	ip_layer_idx_arr_.clear();
	ip_layer_idx_arr_.push_back(num_layers_so_far + c_num_data_split_layers);
	net_def += AddInnerProductStr(cb_ReLU, !cb_Sigmoid, cb_drop, 1, layer_1_nodes, num_layers_so_far, 0.2f);
	ip_layer_idx_arr_.push_back(num_layers_so_far + c_num_data_split_layers);
	net_def += AddInnerProductStr(cb_ReLU, !cb_Sigmoid, cb_drop, 2, layer_2_nodes, num_layers_so_far, 0.2f);
	ip_layer_idx_arr_.push_back(num_layers_so_far + c_num_data_split_layers);
	net_def += AddInnerProductStr(!cb_ReLU, cb_Sigmoid, !cb_drop, 3, 2, num_layers_so_far, 0.0f);
	int loss_layer = num_layers_so_far + 2 - 1; // two layers of split in this config - 1 for zero based index
	net_def += AddSoftmaxAndAccuracyStr(3, num_layers_so_far);
	
	success = google::protobuf::TextFormat::ParseFromString(net_def, net_param);

	net_param = solver_param.add_test_net_param();
	net_def = CreateHD5TestStr("/devlink/github/test/toys/NetGen/GramPosValid/test_list.txt");
	num_layers_so_far = 1;
	net_def += AddInnerProductStr(cb_ReLU, !cb_Sigmoid, cb_drop, 1, layer_1_nodes, num_layers_so_far, 0.2f);
	net_def += AddInnerProductStr(cb_ReLU, !cb_Sigmoid, cb_drop, 2, layer_2_nodes, num_layers_so_far, 0.2f);
	net_def += AddInnerProductStr(!cb_ReLU, cb_Sigmoid, !cb_drop, 3, 2, num_layers_so_far, 0.0f);
	net_def += AddSoftmaxAndAccuracyStr(3, num_layers_so_far);
	success = google::protobuf::TextFormat::ParseFromString(net_def, net_param);
	
//	shared_ptr<caffe::Solver<float> >
//		solver(caffe::GetSolver<float>(solver_param));
	solver_.reset(new NetGenSolver(solver_param));
	solver_->get_net(net_);
	solver_->set_loss_layer(loss_layer);
	ip_layer_idx_idx_ = mod_idx_idx;
	
//	Net<float> * test_net = solver_->get_test_net();
//	HDF5DataLayer<float> * data_layer =  dynamic_cast<HDF5DataLayer<float>*>(test_net->layers()[0].get());
//	data_layer->
	
}

float NGNet::DoRun(bool bIntersection) {
	solver_->reset_loss_sum();
	const double c_highway_run_time = 60.0;
	const double c_intersection_run_time = 30.0;
	double run_time = c_highway_run_time;
	if (bIntersection) {
		run_time = c_intersection_run_time;
	}
	solver_->set_run_time(run_time);
    solver_->Solve();
	//float loss = solver_->get_avg_loss();
	float loss = 0.0f;
	Net<float> * test_net = solver_->get_test_net();
	vector<Blob<float>*> bottom_vec;
	int num_tests = 100000;
	LOG(INFO) << "Testing " << num_tests << " records. \n";
	for (int it = 0; it < num_tests; it++) {
		float iter_loss;
		test_net->Forward(bottom_vec, &iter_loss);
		loss += iter_loss;
	}
	LOG(INFO) << "Testing done. \n";
	
	return loss / (float)num_tests;
	
}

enum ModAction {
	ModActionDoubleLayer1,
	ModActionDoubleLayer2,
	NumModActions 
};
void NetGen::Init() {

//	for (int ig = 0; ig < nets.size(); ig++) {
//		NGNet * net = nets[ig].get();
//	}
	
	vector<int> ip_layer_idx_arr;
	int layer_1_nodes = 10;
	int layer_2_nodes = 3;
	shared_ptr<NGNet> ng_net(new NGNet());
	const int c_num_to_test = 1;
	const int c_max_fails = 10;
	ng_net->Gen(layer_1_nodes, layer_2_nodes, -1);
	int num_fails = 0;
	while(true) {
		float loss_highway = ng_net->DoRun(false);
	
		shared_ptr<NGNet> ng_net_better;
		ModAction better_action;
		bool b_found_one_improvement = false;
		float loss_intersection_change;
		vector<pair<int, ModAction> > phases;
		for (int i_phase = 0; i_phase < NumModActions; i_phase++) {
			phases.push_back(make_pair(rand(), (ModAction)i_phase));
		}
		std::sort(phases.begin(), phases.end());
		for (int ima = 0; ima < c_num_to_test; ima++) {
			shared_ptr<NGNet> ng_net_2;
			ModAction mod_action = phases[ima].second;
			switch(mod_action) {
				case ModActionDoubleLayer1:
					std::cerr << "selecting double layer 1\n";
					ng_net_2.reset(new NGNet());
					ng_net_2->Gen(layer_1_nodes * 2, layer_2_nodes, 0);
					break;
				case ModActionDoubleLayer2:
					std::cerr << "selecting double layer 2\n";
					ng_net_2.reset(new NGNet());
					ng_net_2->Gen(layer_1_nodes, layer_2_nodes * 2, 1);
			}
			ng_net_2->SetWeights(ng_net.get());
			loss_intersection_change = ng_net_2->DoRun(true);
			std::cerr << "change loss went from " << loss_highway << " to " << loss_intersection_change << ". \n";
			if (loss_highway > loss_intersection_change) {
				//ng_net.reset(ng_net_2.get());
				b_found_one_improvement = true;
				ng_net_better = ng_net_2;
				better_action = mod_action;
			}
		} // end pahse loop
		
		float loss_intersection_continue = ng_net->DoRun(true);
		std::cerr << "continue loss went from " << loss_highway << " to " << loss_intersection_continue << ". \n";
		if (b_found_one_improvement && (loss_intersection_change < loss_intersection_continue)) {
			num_fails = 0;
			std::cerr << "upgrading\n";
			switch(better_action) {
				case ModActionDoubleLayer1:
					layer_1_nodes *= 2;
					break;
				case ModActionDoubleLayer2:
					layer_2_nodes *= 2;
					break;
			}			
			ng_net = ng_net_better;
		}
		else if (!b_found_one_improvement && (loss_intersection_continue > loss_highway) ) {
			if (num_fails >= c_max_fails) {
				std::cerr << "Optimiztion complete. No better option\n";
				break;
			}
			num_fails++;
			std::cerr << "Failed to improve, num fails is now " << num_fails << std::endl;
		}
		// third, default option is to keep going with the current net
	}
	
	
	bInit_ = true;

}


/* Return the values in the output layer */
bool NetGen::Classify() {
	CHECK(bInit_) << "NetGen: Init must be called first\n";
	
	

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
  
	
	int input_data_idx = 0;
	int input_label_idx = 1;

	const int mod_param_layer_idx = 2; // 5; // 2
	const int mod_layer_idx = 2; // 4; // 2
	const int mod_layer_idx_post = 4; // 6; // 4

	NetGen generator;
	vector<shared_ptr<NGNet> > nets;
	generator.Init();

	
}
#endif // CAFFE_MULTINET_MAIN