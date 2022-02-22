#include "paddle/include/paddle_inference_api.h"
#include <chrono>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <fstream>
#include <thread>
#include <mutex>

DEFINE_bool(single_instance, false, "Each model only has one predictor");
DEFINE_bool(single_thread, false, "Test all model one by one in main thread");
DEFINE_string(test_groups, "-1", "Set test group models, -1 means test all model");
DEFINE_int32(sample_max_num, 20, "Set the max num of input samples for every model,"
                                 " -1 means use all input samples");

using namespace paddle_infer;
using Time = decltype(std::chrono::high_resolution_clock::now());

Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

struct Params {
  std::string model_path;
  std::string input_path;
  int is_int8_model;
  int instance;
  int num_threads;
  int use_mkldnn;
  int enable_ir_optim;
};

struct InputData {
  int channels;
  int height;
  int width;
  int batch_size;
  std::vector<float> input;
};

void ReadTxtData(const std::string& path, std::vector<InputData> &input_data_list) {
  std::ifstream fin(path, std::ifstream::in);
  int num = 0;
  if (fin.is_open()) {
      std::string str;
      while (getline(fin, str)) {
          InputData input_data;

          std::stringstream stream(str);
          stream >> input_data.batch_size;
          stream >> input_data.channels;
          stream >> input_data.height;
          stream >> input_data.width;

          std::vector<float> &input = input_data.input;
          std::string str1;
          getline(fin, str1);
          std::stringstream stream1(str1);
          float a;
          while(stream1>>a) {
            input.push_back(a);
          }

          input_data_list.emplace_back(input_data);

          num++;
          if (FLAGS_sample_max_num > 0 && num >= FLAGS_sample_max_num) {
            break;
          }
      }
      fin.close();
  } else {
      LOG(FATAL) << "ReadTxtData error";
      return;
  }
}

std::shared_ptr<Predictor> CreatePredictor(const Params& params) {
  Config config;
//config.SetModel(params.model_path + "/model", params.model_path + "/params");
  config.SetModel(params.model_path);

  config.DisableGpu();
  if (params.use_mkldnn) {
    config.EnableMKLDNN();
    LOG(INFO) << "enable mkldnn";
  }

  config.SetCpuMathLibraryNumThreads(params.num_threads);
  LOG(INFO) << "threads:" << params.num_threads;

  config.EnableMemoryOptim();
  config.SwitchUseFeedFetchOps(false);

  config.DisableGlogInfo();
  //_config.SwitchIrDebug(true);

  if (params.enable_ir_optim) {
    config.SwitchIrOptim(true);
    LOG(INFO) << "enable ir optim";
  } else {
    config.SwitchIrOptim(false);
    LOG(INFO) << "disable ir optim";
  }

  std::shared_ptr<Predictor> predictor = CreatePredictor(config);
  return predictor;
}

std::shared_ptr<Predictor> CreatePredictor(const std::string& model_path, 
  bool is_int8_model, int threads) {

  Config config;
  config.SetModel(model_path + "/model", model_path + "/params");

  config.DisableGpu();
  config.EnableMKLDNN();
  config.SetCpuMathLibraryNumThreads(threads);

  config.EnableMemoryOptim();
  config.SwitchUseFeedFetchOps(false);

  config.DisableGlogInfo();
  //_config.SwitchIrDebug(true);

  if (is_int8_model) {
    config.SwitchIrOptim(false);
    LOG(INFO) << "Disable ir opt";
  } else {
    config.SwitchIrOptim(true);
    LOG(INFO) << "Enable ir opt";
  }

  std::shared_ptr<Predictor> predictor = CreatePredictor(config);
  return predictor;
}

void Predict(std::shared_ptr<Predictor> predictor, const std::string& input_path,
             int epoch, const std::string& model_name, int model_index) {
  std::string model_rep = model_name + " " + std::to_string(model_index);
  LOG(INFO) << "Start predictiton, " << model_rep;

  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);

  // set input
//  InputData& input_data = input_data_list[num];
  int batch_size = 128;
  int channels = 3;
  int height = 224;
  int width = 224;

  input_t->Reshape({batch_size, channels, height, width});

  int nums = batch_size * channels * height * width;
  std::vector<float> input(nums, 0);

  input_t->CopyFromCpu(input.data());

  //warmup
  //LOG(INFO) << "This i for warmup only" ;
  for (auto iter = 0; iter<1; iter++){
    CHECK(predictor->Run());
    LOG(INFO) <<"Warmup " << iter << " batches";
  }
  // run epoch * input_num times of inference  
  double cost_time = 0;
  auto input_num=20;
  for (int num = 0; num < input_num; ++num) {
    // run
    auto time1 = time();
    CHECK(predictor->Run());
    auto time2 = time();
    cost_time += time_diff(time1, time2);
  }
  LOG(INFO) << "Finish prediction, FPS: " << input_num*batch_size/(cost_time/1000.0); 

  LOG(INFO) << "Finish predictiton, " << model_rep
//          << ", repeat:" << repeat_idx
          << ", avg time:" << cost_time / (input_num*batch_size) << " ms";
  LOG(INFO) << "-------Finish all prediction, " << model_rep;
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  /*
  struct Params {
    std::string model_path;
    std::string input_path;
    int is_int8_model;
    int instance;
    int num_threads;
    int use_mkldnn;
    int enable_ir_optim;
  };
  */
  std::vector<std::vector<Params>> all_model_params = {
    // trainticket 0
    {
      {"/home/danqing/PP_workload/PaddleSlim/demo/mkldnn_quant/resnet50_int8", "input/trainticket/trainticket_line_recg_input_layer_0.txt", 1, 28,4,1,1},
    },
  };

  auto get_model_name = [](const std::string& model_path) {
    auto first = model_path.find_first_of("/");
    auto second = model_path.find_first_of("/", first + 1);
    auto third = model_path.find_first_of("/", second + 1);
    return model_path.substr(first + 1, third - first - 1);
  };

  std::vector<Params> test_model_params = {};
  int epoch = 20; 

  // get inputs
  if (test_model_params.empty()) {
    std::stringstream ss(FLAGS_test_groups);
    int tmp_int = 0;
    ss>>tmp_int;
    if(tmp_int==0){
	for (auto& item: all_model_params){
		test_model_params.push_back(item[0]);
	}	
    } 
  }
  
  // test all model one by one in main thread
  if (FLAGS_single_thread) {
    for (int i = 0; i < test_model_params.size(); i++) {
      auto& param = test_model_params[i];
      auto predictor = CreatePredictor(param);

      auto model_name = get_model_name(param.model_path);
      Predict(predictor, param.input_path, epoch, model_name, i);
    }
    return 0;
  }

  // create predictors
  std::vector<std::thread> run_threads;
  std::vector<std::shared_ptr<Predictor>> predictor_list;
  std::vector<Params> tmp_model_params;
  // test_model_params.size() = test_group * each_group_num
  // predictor_list_size: test_model_params.size() * param.instance 

  LOG(INFO) << "Create predictor.";
  for (int i = 0; i < test_model_params.size(); i++) {
    auto& param = test_model_params[i];

    auto predictor = CreatePredictor(param);
    
    predictor_list.push_back(predictor);
    tmp_model_params.push_back(param);

    for (int j = 1; j < param.instance && !FLAGS_single_instance; j++) {
      predictor_list.emplace_back(std::move(predictor->Clone()));
      tmp_model_params.push_back(param);
    }
  }
  LOG(INFO) << "Predictor num:" << predictor_list.size() << "\n";
  
  // run predictors

  for (int i = 0; i < predictor_list.size(); i++) {
    auto& param = tmp_model_params[i];
    std::string input_path = param.input_path;
    std::string model_name = get_model_name(param.model_path);

    run_threads.emplace_back(Predict, predictor_list[i],
      input_path, epoch, model_name, i);
  }

  LOG(INFO) << "Run model \n";
  for (int i = 0; i < run_threads.size(); i++) {
    run_threads[i].join();
  }
}
