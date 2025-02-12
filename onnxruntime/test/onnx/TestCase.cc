// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// needs to be included first to get around onnxruntime\cmake\external\onnx\onnx/common/constants.h(14): error C2513: 'bool': no variable declared before '='

#include "TestCase.h"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <map>
#include <regex>
#include <set>
#include <string>

#include "callback.h"
#include "heap_buffer.h"
#include "mem_buffer.h"
#include "onnx_model_info.h"
#include "pb_helper.h"
#include "tensorprotoutils.h"

#include "core/common/logging/logging.h"
#include "core/common/common.h"
#include "core/platform/env.h"
#include <mutex>
#include "fnmatch_simple.h"
#include "core/platform/path_lib.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/allocator.h"
#include "core/framework/TensorSeq.h"
#include "re2/re2.h"

using namespace onnxruntime;
using namespace onnxruntime::common;

static constexpr int protobuf_block_size_in_bytes = 4 * 1024 * 1024;

const std::string TestModelInfo::unknown_version = "unknown version";

namespace {
using PATH_STRING_TYPE = std::basic_string<PATH_CHAR_TYPE>;

static int ExtractFileNo(const std::filesystem::path& pathstr) {
  PATH_STRING_TYPE name = pathstr;
  size_t p1 = name.rfind('.');
  size_t p2 = name.rfind('_', p1);
  ++p2;
  PATH_STRING_TYPE number_str = name.substr(p2, p1 - p2);
  const PATH_CHAR_TYPE* start = number_str.c_str();
  const PATH_CHAR_TYPE* end = start;
  long ret = OrtStrtol(start, const_cast<PATH_CHAR_TYPE**>(&end));
  if (end == start) {
    ORT_THROW("parse file name failed");
  }
  return static_cast<int>(ret);
}

static void SortFileNames(std::vector<std::filesystem::path>& input_pb_files) {
  if (input_pb_files.size() <= 1) return;
  std::sort(input_pb_files.begin(), input_pb_files.end(),
            [](const std::filesystem::path& left, std::filesystem::path& right) -> bool {
              int left1 = ExtractFileNo(left.filename());
              int right1 = ExtractFileNo(right.filename());
              return left1 < right1;
            });

  for (size_t i = 0; i != input_pb_files.size(); ++i) {
    int fileno = ExtractFileNo(GetLastComponent(input_pb_files[i]));
    if (static_cast<size_t>(fileno) != i) {
      std::basic_ostringstream<PATH_CHAR_TYPE> oss;
      oss << input_pb_files[0];
      for (size_t j = 1; j != input_pb_files.size(); ++j)
        oss << ORT_TSTR(" ") << input_pb_files[j];
      ORT_THROW("illegal input file name:", ToUTF8String(oss.str()));
    }
  }
}

}  // namespace
#if !defined(ORT_MINIMAL_BUILD)
std::unique_ptr<TestModelInfo> TestModelInfo::LoadOnnxModel(const std::filesystem::path& model_url) {
  return std::make_unique<TestModelInfo>(model_url);
}
#endif

std::unique_ptr<TestModelInfo> TestModelInfo::LoadOrtModel(const std::filesystem::path& model_url) {
  return std::make_unique<TestModelInfo>(model_url, true);
}

/**
 * test_case_dir must have contents of:
 * model.onnx
 * ???/input_??.pb
 * ???/output_??.pb
 * ???/input_??.pb
 * ???/output_??.pb
 */
class OnnxTestCase : public ITestCase {
 private:
  std::string test_case_name_;
  mutable std::vector<std::string> debuginfo_strings_;
  mutable std::mutex m_;

  std::vector<std::filesystem::path> test_data_dirs_;

  std::string GetDatasetDebugInfoString(size_t dataset_id) const override {
    std::lock_guard<std::mutex> l(m_);
    if (dataset_id < debuginfo_strings_.size()) {
      return debuginfo_strings_[dataset_id];
    }
    // return empty string
    return std::string();
  }

  void ConvertTestData(const ONNX_NAMESPACE::TensorProto& test_data_pb,
                       onnxruntime::test::HeapBuffer& b,
                       bool is_input, size_t i,
                       std::unordered_map<std::string, Ort::Value>& out) const;

  void ConvertTestData(const ONNX_NAMESPACE::SequenceProto& test_data_pb,
                       onnxruntime::test::HeapBuffer& b,
                       bool is_input, size_t i,
                       std::unordered_map<std::string, Ort::Value>& out) const;

#if !defined(DISABLE_OPTIONAL_TYPE)
  void ConvertTestData(const ONNX_NAMESPACE::OptionalProto& test_data_pb,
                       onnxruntime::test::HeapBuffer& b,
                       bool is_input, size_t i,
                       std::unordered_map<std::string, Ort::Value>& out) const;
#endif

  std::once_flag model_parsed_;
  std::once_flag config_parsed_;
  double per_sample_tolerance_;
  double relative_per_sample_tolerance_;
  bool post_processing_;
  std::unique_ptr<TestModelInfo> model_info_;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OnnxTestCase);

 public:
  OnnxTestCase(const std::string& test_case_name, _In_ std::unique_ptr<TestModelInfo> model,
               double default_per_sample_tolerance, double default_relative_per_sample_tolerance);
  void GetPerSampleTolerance(double* value) const override;
  void GetRelativePerSampleTolerance(double* value) const override;
  void GetPostProcessing(bool* value) const override;

  const ONNX_NAMESPACE::ValueInfoProto* GetInputInfoFromModel(size_t i) const override {
    return model_info_->GetInputInfoFromModel(i);
  }

  const ONNX_NAMESPACE::ValueInfoProto* GetOutputInfoFromModel(size_t i) const override {
    return model_info_->GetOutputInfoFromModel(i);
  }

  size_t GetDataCount() const override { return test_data_dirs_.size(); }
  const std::string& GetNodeName() const override { return model_info_->GetNodeName(); }
  const std::filesystem::path& GetModelUrl() const override { return model_info_->GetModelUrl(); }
  const std::string& GetTestCaseName() const override { return test_case_name_; }
  std::string GetTestCaseVersion() const override { return model_info_->GetNominalOpsetVersion(); }

  void LoadTestData(size_t id, onnxruntime::test::HeapBuffer& b, std::unordered_map<std::string, Ort::Value>&,
                    bool is_input) const override;
};

std::unique_ptr<ITestCase> CreateOnnxTestCase(const std::string& test_case_name,
                                              std::unique_ptr<TestModelInfo> model,
                                              double default_per_sample_tolerance,
                                              double default_relative_per_sample_tolerance) {
  return std::make_unique<OnnxTestCase>(test_case_name, std::move(model),
                                        default_per_sample_tolerance,
                                        default_relative_per_sample_tolerance);
}

void OnnxTestCase::GetPerSampleTolerance(double* value) const {
  *value = per_sample_tolerance_;
}

void OnnxTestCase::GetRelativePerSampleTolerance(double* value) const {
  *value = relative_per_sample_tolerance_;
}

void OnnxTestCase::GetPostProcessing(bool* value) const {
  *value = post_processing_;
}

// CentOS lacks find_if
template <class Iter, class Pred>
inline Iter find_with_pred(Iter first, Iter last, Pred p) {
  while (first != last) {
    if (p(*first)) {
      break;
    }
    ++first;
  }
  return first;
}

static std::string trim_str(const std::string& in) {
  std::string s = in;
  s.erase(s.begin(), find_with_pred(s.begin(), s.end(), [](int ch) {
            return !std::isspace(ch);
          }));
  s.erase(find_with_pred(s.rbegin(), s.rend(), [](int ch) {
            return !std::isspace(ch);
          }).base(),
          s.end());
  return s;
}

/**
 * @brief Read a text file that each line is a key value pair separated by ':'
 * @param path File path
 * @param fc output key value pairs
 * @return True, success. False, the file doesn't exist or could be read.
 */
static bool ReadConfigFile(const std::filesystem::path& path, std::map<std::string, std::string>& fc) {
  if (!std::filesystem::exists(path)) return false;
  std::ifstream infile(path);
  if (!infile.good()) {
    return false;
  }

  for (std::string line; std::getline(infile, line);) {
    std::istringstream ss(line);
    if (line.empty()) {
      continue;
    }
    std::vector<std::string> tokens;
    for (std::string token; std::getline(ss, token, ':');) {
      std::string trimmed_token = trim_str(token);
      if (trimmed_token.empty()) {
        continue;
      }
      tokens.push_back(trimmed_token);
    }
    fc[tokens[0]] = tokens[1];
  }
  return true;
}

// load tensors from disk
template <typename PATH_STRING_TYPE>
static void LoadTensor(const PATH_STRING_TYPE& pb_file, ONNX_NAMESPACE::TensorProto& input_pb) {
  int tensor_fd;
  auto st = Env::Default().FileOpenRd(pb_file, tensor_fd);
  if (!st.IsOK()) {
    ORT_THROW("open file '", ToUTF8String(pb_file), "' failed:", st.ErrorMessage());
  }
  google::protobuf::io::FileInputStream f(tensor_fd, protobuf_block_size_in_bytes);
  f.SetCloseOnDelete(true);
  if (!input_pb.ParseFromZeroCopyStream(&f)) {
    ORT_THROW("parse file '", ToUTF8String(pb_file), "' failed");
  }
}

// load sequence tensors from disk
template <typename PATH_STRING_TYPE>
static void LoadSequenceTensor(const PATH_STRING_TYPE& pb_file, ONNX_NAMESPACE::SequenceProto& input_pb) {
  int tensor_fd;
  auto st = Env::Default().FileOpenRd(pb_file, tensor_fd);
  if (!st.IsOK()) {
    ORT_THROW("open file '", ToUTF8String(pb_file), "' failed:", st.ErrorMessage());
  }
  google::protobuf::io::FileInputStream f(tensor_fd, protobuf_block_size_in_bytes);
  f.SetCloseOnDelete(true);
  if (!input_pb.ParseFromZeroCopyStream(&f)) {
    ORT_THROW("parse file '", ToUTF8String(pb_file), "' failed");
  }
}

#if !defined(DISABLE_OPTIONAL_TYPE)
template <typename PATH_STRING_TYPE>
static void LoadOptional(const PATH_STRING_TYPE& pb_file,
                         ONNX_NAMESPACE::OptionalProto& input_pb) {
  int tensor_fd;
  auto st = Env::Default().FileOpenRd(pb_file, tensor_fd);
  if (!st.IsOK()) {
    ORT_THROW("open file '", ToUTF8String(pb_file), "' failed:", st.ErrorMessage());
  }
  google::protobuf::io::FileInputStream f(tensor_fd, protobuf_block_size_in_bytes);
  f.SetCloseOnDelete(true);
  if (!input_pb.ParseFromZeroCopyStream(&f)) {
    ORT_THROW("parse file '", ToUTF8String(pb_file), "' failed");
  }
}
#endif

void OnnxTestCase::LoadTestData(size_t id, onnxruntime::test::HeapBuffer& b,
                                std::unordered_map<std::string, Ort::Value>& name_data_map,
                                bool is_input) const {
  if (id >= test_data_dirs_.size()) {
    ORT_THROW("index out of bound");
  }

  std::vector<std::filesystem::path> test_data_pb_files = SimpleGlob(test_data_dirs_[id], is_input ? ORT_TSTR("input_*.pb") : ORT_TSTR("output_*.pb"));


  SortFileNames(test_data_pb_files);

  for (size_t i = 0; i < test_data_pb_files.size(); ++i) {
    const ONNX_NAMESPACE::ValueInfoProto* value_info_proto = is_input ? model_info_->GetInputInfoFromModel(i) : model_info_->GetOutputInfoFromModel(i);
    if (!value_info_proto->has_type()) {
      ORT_THROW("Model ", is_input ? "input " : "output ", i, " is missing type info");
    }

    if (value_info_proto->type().has_tensor_type()) {
      ONNX_NAMESPACE::TensorProto test_pb;
      LoadTensor(test_data_pb_files[i], test_pb);
      ConvertTestData(test_pb, b, is_input, i, name_data_map);
    } else if (value_info_proto->type().has_sequence_type()) {
      ONNX_NAMESPACE::SequenceProto test_pb;
      LoadSequenceTensor(test_data_pb_files[i], test_pb);
      ConvertTestData(test_pb, b, is_input, i, name_data_map);
    }
#if !defined(DISABLE_OPTIONAL_TYPE)
    else if (value_info_proto->type().has_optional_type()) {
      ONNX_NAMESPACE::OptionalProto test_pb;
      LoadOptional(test_data_pb_files[i], test_pb);
      ConvertTestData(test_pb, b, is_input, i, name_data_map);
    }
#endif
    else {
      ORT_THROW("Unsupported type for the ", is_input ? "input " : "output ", i, " in the test runner");
    }
  }
}

void OnnxTestCase::ConvertTestData(const ONNX_NAMESPACE::TensorProto& test_data_pb,
                                   onnxruntime::test::HeapBuffer& b,
                                   bool is_input, size_t i,
                                   std::unordered_map<std::string, Ort::Value>& out) const {
  const std::string& name = test_data_pb.name();
  const std::string& name_finalized = !name.empty()
                                          ? name
                                          : (is_input ? model_info_->GetInputName(i) : model_info_->GetOutputName(i));

  size_t len = 0;

  auto status = onnxruntime::test::GetSizeInBytesFromTensorProto<0>(test_data_pb, &len);
  if (!status.IsOK()) {
    ORT_THROW(status.ToString());
  }
  void* p = len == 0 ? nullptr : b.AllocMemory(len);
  Ort::Value v1{nullptr};
  onnxruntime::test::OrtCallback d;
  OrtMemoryInfo cpu_memory_info(onnxruntime::CPU, OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeDefault);
  status = onnxruntime::test::TensorProtoToMLValue(test_data_pb, onnxruntime::test::MemBuffer(p, len, cpu_memory_info),
                                                   v1, d);
  if (!status.IsOK()) {
    ORT_THROW(status.ToString());
  }
  if (d.f) {
    b.AddDeleter(d);
  }
  out.emplace(name_finalized, std::move(v1));
}

void OnnxTestCase::ConvertTestData(const ONNX_NAMESPACE::SequenceProto& test_data_pb,
                                   onnxruntime::test::HeapBuffer& b,
                                   bool is_input, size_t i,
                                   std::unordered_map<std::string, Ort::Value>& out) const {
  const std::string& name = test_data_pb.name();
  const std::string& name_finalized = !name.empty()
                                          ? name
                                          : (is_input ? model_info_->GetInputName(i) : model_info_->GetOutputName(i));

  size_t len = 0;

  std::vector<Ort::Value> seq;
  if (test_data_pb.elem_type() != ONNX_NAMESPACE::SequenceProto_DataType_TENSOR) {
    ORT_THROW("Only parsing a sequence of tensors is currently supported");
  }
  const auto& tensors = test_data_pb.tensor_values();
  const size_t val = tensors.size();
  seq.reserve(val);

  for (auto it = tensors.cbegin(); it != tensors.cend(); ++it) {
    auto status = onnxruntime::test::GetSizeInBytesFromTensorProto<0>(*it, &len);
    if (!status.IsOK()) {
      ORT_THROW(status.ToString());
    }
    void* p = len == 0 ? nullptr : b.AllocMemory(len);
    Ort::Value v1{nullptr};
    onnxruntime::test::OrtCallback d;
    OrtMemoryInfo cpu_memory_info(onnxruntime::CPU, OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeDefault);
    status = onnxruntime::test::TensorProtoToMLValue(*it, onnxruntime::test::MemBuffer(p, len, cpu_memory_info),
                                                     v1, d);
    if (!status.IsOK()) {
      ORT_THROW(status.ToString());
    }
    if (d.f) {
      b.AddDeleter(d);
    }

    seq.push_back(std::move(v1));
  }

  if (seq.size() == 0) {
    // TODO: implement support for creating empty sequences. Not urgent yet since we don't have real world models.
    // For now, only the single node ONNX test - `test_loop13_seq` requires it (will keep it disabled for now).
    ORT_THROW("Creation of empty sequences is currently not supported in the test runner");
  } else {
    out.emplace(name_finalized, Ort::Value::CreateSequence(seq));
  }
}

#if !defined(DISABLE_OPTIONAL_TYPE)
void OnnxTestCase::ConvertTestData(const ONNX_NAMESPACE::OptionalProto& test_data_pb,
                                   onnxruntime::test::HeapBuffer& b,
                                   bool is_input, size_t i,
                                   std::unordered_map<std::string, Ort::Value>& out) const {
  // Optional Tensor
  if (test_data_pb.elem_type() ==
      ONNX_NAMESPACE::OptionalProto_DataType::OptionalProto_DataType_TENSOR) {
    // The optional tensor is not "None", deal with it as a regular tensor
    if (test_data_pb.has_tensor_value()) {
      ConvertTestData(test_data_pb.tensor_value(), b, is_input, i, out);
    } else {
      // Process None
      // If is_input is true, don't include the None in the feeds
      // If is_input is false, include it in the fetches, so that we can validate
      // whether we received a None output from ORT.

      if (!is_input) {
        const std::string& name = test_data_pb.name();
        const std::string& name_finalized = !name.empty()
                                                ? name
                                                : (is_input ? model_info_->GetInputName(i) : model_info_->GetOutputName(i));

        // Our API doesn't support creating None OrtValue,
        // so we place an nullptr into the expected values.
        Ort::Value value{nullptr};
        out.emplace(name_finalized, std::move(value));
      }
    }
  }  // Optional Sequence Tensor
  else if (test_data_pb.elem_type() ==
           ONNX_NAMESPACE::OptionalProto_DataType::OptionalProto_DataType_SEQUENCE) {
    // The optional sequence tensor is not "None", deal with it as a regular tensor
    if (test_data_pb.has_sequence_value()) {
      // ConvertTestData() ensures that sequence contains only tensors - we do no need
      // a redundant check here
      ConvertTestData(test_data_pb.sequence_value(), b, is_input, i, out);
    } else {
      // Process None
      // If is_input is true, don't include the None in the feeds
      // If is_input is false, include it in the fetches, so that we can validate
      // whether we received a None output from ORT.

      if (!is_input) {
        const std::string& name = test_data_pb.name();
        const std::string& name_finalized = !name.empty()
                                                ? name
                                                : (is_input ? model_info_->GetInputName(i) : model_info_->GetOutputName(i));

        // Our API doesn't support creating None OrtValue,
        // so we place an nullptr into the expected values.
        Ort::Value value{nullptr};
        out.emplace(name_finalized, std::move(value));
      }
    }
  }
}
#endif

OnnxTestCase::OnnxTestCase(const std::string& test_case_name, _In_ std::unique_ptr<TestModelInfo> model,
                           double default_per_sample_tolerance, double default_relative_per_sample_tolerance)
    : test_case_name_(test_case_name), model_info_(std::move(model)) {
  std::filesystem::path test_case_dir = model_info_->GetDir();
  if (!std::filesystem::exists(test_case_dir)) {
    ORT_THROW("test case dir doesn't exist");
  }
  // parse config
  std::filesystem::path config_path =
      test_case_dir / ORT_TSTR("config.txt");
  /* Note: protobuf-lite doesn't support reading protobuf files as text-format. Config.txt is exactly that.
     That's the reason I've to parse the file in a different way to read the configs. Currently
     this affects 2 tests - fp16_tiny_yolov2 and fp16_inception_v1. It's not clear why we've to use protobuf
     to represent simple config files that have only key-value pairs.
   */
  std::map<std::string, std::string> fc;
  per_sample_tolerance_ = default_per_sample_tolerance;
  relative_per_sample_tolerance_ = default_relative_per_sample_tolerance;
  post_processing_ = false;
  if (ReadConfigFile(config_path, fc)) {
    if (fc.count("per_sample_tolerance") > 0) {
      per_sample_tolerance_ = stod(fc["per_sample_tolerance"]);
    }
    if (fc.count("relative_per_sample_tolerance") > 0) {
      relative_per_sample_tolerance_ = stod(fc["relative_per_sample_tolerance"]);
    }
    if (fc.count("post_processing") > 0) {
      post_processing_ = fc["post_processing"] == "true";
    }
  }
  for (auto const& dir_entry : std::filesystem::directory_iterator(test_case_dir)) {
    if (!dir_entry.is_directory()) continue;
    test_data_dirs_.push_back(dir_entry.path());
    debuginfo_strings_.push_back(ToUTF8String(dir_entry.path().string()));
  }
}

bool IsValidTest(std::basic_string<PATH_CHAR_TYPE> test_case_name, const std::vector<std::basic_string<PATH_CHAR_TYPE>>& whitelisted_test_cases, const std::unordered_set<std::basic_string<ORTCHAR_T>>& disabled_tests) {
  if (test_case_name.compare(0, 5, ORT_TSTR("test_")) == 0) test_case_name = test_case_name.substr(5);

  if (!whitelisted_test_cases.empty() && std::find(whitelisted_test_cases.begin(), whitelisted_test_cases.end(),
                                                   test_case_name) == whitelisted_test_cases.end()) {
    return false;
  }
  return disabled_tests.find(test_case_name) == disabled_tests.end();
}

void LoadSingleModel(std::unique_ptr<TestModelInfo> model_info, const TestTolerances& tolerances, std::unique_ptr<std::set<BrokenTest>>& broken_tests,
                     std::unique_ptr<std::set<std::string>>& broken_tests_keyword_set,
                     const std::function<void(std::unique_ptr<ITestCase>)>& process_function) {
  auto test_case_dir = model_info->GetDir();
  auto test_case_name = test_case_dir.filename().native();
  if (test_case_name.compare(0, 5, ORT_TSTR("test_")) == 0) test_case_name = test_case_name.substr(5);
  auto test_case_name_in_log = test_case_name + ORT_TSTR(" in ") + test_case_dir.native();

#if !defined(ORT_MINIMAL_BUILD) && !defined(USE_QNN) && !defined(USE_VSINPU)
  // to skip some models like *-int8 or *-qdq
  if ((reinterpret_cast<TestModelInfo*>(model_info.get()))->HasDomain(ONNX_NAMESPACE::AI_ONNX_TRAINING_DOMAIN) ||
      (reinterpret_cast<TestModelInfo*>(model_info.get()))->HasDomain(ONNX_NAMESPACE::AI_ONNX_PREVIEW_TRAINING_DOMAIN)) {
    fprintf(stderr, "Skip test case:: %s %s\n", ToUTF8String(test_case_name_in_log).c_str(), " as it has training domain");
    return;
  }
#endif

  if (broken_tests) {
    BrokenTest t = {ToUTF8String(test_case_name), ""};
    auto iter = broken_tests->find(t);
    auto opset_version = model_info->GetNominalOpsetVersion();
    if (iter != broken_tests->end() &&
        (opset_version == TestModelInfo::unknown_version || iter->broken_opset_versions_.empty() ||
         iter->broken_opset_versions_.find(opset_version) != iter->broken_opset_versions_.end())) {
      fprintf(stderr, "Skip test case:: %s %s\n", ToUTF8String(test_case_name_in_log).c_str(), " due to broken_tests");
      return;
    }
  }

  if (broken_tests_keyword_set) {
    for (auto iter2 = broken_tests_keyword_set->begin(); iter2 != broken_tests_keyword_set->end(); ++iter2) {
      std::string keyword = *iter2;
      if (ToUTF8String(test_case_name).find(keyword) != std::string::npos) {
        fprintf(stderr, "Skip test case:: %s %s\n", ToUTF8String(test_case_name_in_log).c_str(), " as it is in broken test keywords");
        return;
      }
    }
  }

  const auto tolerance_key = ToUTF8String(test_case_dir.filename());

  std::unique_ptr<ITestCase> l = CreateOnnxTestCase(ToUTF8String(test_case_name), std::move(model_info),
                                                    tolerances.absolute(tolerance_key),
                                                    tolerances.relative(tolerance_key));
  fprintf(stdout, "Load Test Case: %s\n", ToUTF8String(test_case_name_in_log).c_str());
  process_function(std::move(l));
}

void LoadTests(const std::vector<std::basic_string<PATH_CHAR_TYPE>>& input_paths,
               const std::vector<std::basic_string<PATH_CHAR_TYPE>>& whitelisted_test_cases,
               const TestTolerances& tolerances,
               const std::unordered_set<std::basic_string<ORTCHAR_T>>& disabled_tests,
               std::unique_ptr<std::set<BrokenTest>> broken_tests,
               std::unique_ptr<std::set<std::string>> broken_tests_keyword_set,
               const std::function<void(std::unique_ptr<ITestCase>)>& process_function) {
  std::vector<std::filesystem::path> onnx_models;
  std::vector<std::filesystem::path> ort_models;
  for (const std::basic_string<PATH_CHAR_TYPE>& path_str : input_paths) {
    ORT_TRY {
      for (auto& dir_entry : std::filesystem::recursive_directory_iterator(path_str)) {
        if (!dir_entry.is_regular_file() || dir_entry.is_directory()) continue;
        std::filesystem::path node_data_root_path = dir_entry.path();
        std::filesystem::path filename_str = dir_entry.path().filename();
        if (filename_str.empty() || filename_str.native()[0] == ORT_TSTR('.')) {
          // Ignore hidden files.
          continue;
        }
        auto folder_path = node_data_root_path.parent_path().native();
        if (FnmatchSimple(ORT_TSTR("*.onnx"), filename_str.native()) && IsValidTest(folder_path, whitelisted_test_cases, disabled_tests)) {
          onnx_models.push_back(node_data_root_path);
        } else if (FnmatchSimple(ORT_TSTR("*.ort"), filename_str.native()) && IsValidTest(folder_path, whitelisted_test_cases, disabled_tests)) {
          ort_models.push_back(node_data_root_path);
        }
      }
    }
    ORT_CATCH(const std::filesystem::filesystem_error&) {
      // silently ignore the directories that do not exist
    }
  }

#if !defined(ORT_MINIMAL_BUILD)
  // The for-loop below needs to load every ONNX model into memory then destory the in-memory objects, which is very inefficient since 1. in total we need to load every model twice 2. at here we do the job sequentially. 
  // Originally the design was to make the TestModelInfo lightweight so that all the model information can be retrieved from filesystem meta data without actually loading the models.
  for (const std::filesystem::path& model_path : onnx_models) {
    LoadSingleModel(TestModelInfo::LoadOnnxModel(model_path), tolerances, broken_tests, broken_tests_keyword_set, process_function);
  }
#endif
  for (const std::filesystem::path& model_path : ort_models) {
    LoadSingleModel(TestModelInfo::LoadOrtModel(model_path), tolerances, broken_tests, broken_tests_keyword_set, process_function);
  }
}

TestTolerances::TestTolerances(
    double absolute_default, double relative_default,
    const Map& absolute_overrides,
    const Map& relative_overrides) : absolute_default_(absolute_default),
                                     relative_default_(relative_default),
                                     absolute_overrides_(absolute_overrides),
                                     relative_overrides_(relative_overrides) {}

double TestTolerances::absolute(const std::string& name) const {
  const auto iter = absolute_overrides_.find(name);
  if (iter == absolute_overrides_.end()) {
    return absolute_default_;
  }
  return iter->second;
}

double TestTolerances::relative(const std::string& name) const {
  const auto iter = relative_overrides_.find(name);
  if (iter == relative_overrides_.end()) {
    return relative_default_;
  }
  return iter->second;
}

std::unique_ptr<std::set<BrokenTest>> GetBrokenTests(const std::string& provider_name) {
  auto broken_tests = std::make_unique<std::set<BrokenTest>>(std::initializer_list<BrokenTest>{
      {"slice_neg_steps",
       "Type parameter (Tind) bound to different types (tensor(int64) and tensor(int32) in node ()."},
      {"cast_BFLOAT16_to_FLOAT", "Unexpected input data type"},
      {"loop13_seq", "Creation of empty sequences is currently not supported in the test runner"},
      {"sequence_insert_at_front", "shape mismatch, expect {4} got {3}"},
      {"cast_FLOAT_to_BFLOAT16", "expect uint16 got bfloat16"},
      {"mnist", "Input data isn't in valid range"},
      {"BERT_Squad", "test data bug"},
      {"constantofshape_float_ones", "test data bug", {"opset9", "opset10"}},
      {"constantofshape_int_zeros", "test data bug", {"opset9", "opset10"}},
      {"cast_STRING_to_FLOAT", "Linux CI has old ONNX python package with bad test data", {"opset9", "opset10"}},
      // Numpy float to string has unexpected rounding for some results given numpy default precision is meant to be 8.
      // "e.g. 0.296140194 -> '0.2961402' not '0.29614019'. ORT produces the latter with precision set to 8,
      // which doesn't match the expected output that was generated with numpy.
      {"cast_FLOAT_to_STRING", "Numpy float to string has unexpected rounding for some results."},
      {"tf_nasnet_large", "disable temporarily"},
      {"tf_nasnet_mobile", "disable temporarily"},
      {"tf_pnasnet_large", "disable temporarily"},
      {"shrink", "test case is wrong", {"opset9"}},
      {"maxpool_with_argmax_2d_precomputed_strides", "ShapeInferenceError"},
      {"tf_inception_v2", "result mismatch"},
      {"tf_resnet_v1_50", "result mismatch when Conv BN Fusion is applied"},
      {"tf_resnet_v1_101", "result mismatch when Conv BN Fusion is applied"},
      {"tf_resnet_v1_152", "result mismatch when Conv BN Fusion is applied"},
      {"mxnet_arcface", "Model is an invalid ONNX model"},
      {"unique_not_sorted_without_axis", "Expected data for 'Y' is incorrect and in sorted order."},
      {"cumsum_1d_reverse_exclusive", "only failing linux GPU CI. Likely build error."},
      {"resize_downsample_scales_cubic_align_corners", "results mismatch with onnx tests"},
      {"resize_downsample_scales_linear_align_corners", "results mismatch with onnx tests"},
      {"resize_tf_crop_and_resize", "Bad onnx test output. Needs test fix."},
      {"resize_upsample_sizes_nearest_ceil_half_pixel", "Bad onnx test output. Needs test fix."},
      {"resize_upsample_sizes_nearest_floor_align_corners", "Bad onnx test output. Needs test fix."},
      {"resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric", "Bad onnx test output. Needs test fix."},
      {"bitshift_right_uint16", "BitShift(11) uint16 support not enabled currently"},
      {"bitshift_left_uint16", "BitShift(11) uint16 support not enabled currently"},
      {"maxunpool_export_with_output_shape",
       "Invalid output in ONNX test. See https://github.com/onnx/onnx/issues/2398"},
      {"cntk_simple_seg", "Bad onnx test output caused by wrong SAME_UPPER/SAME_LOWER for ConvTranspose"},
      {"training_dropout", "result differs", {}},               // Temporary, subsequent PR will remove this.
      {"training_dropout_default", "result differs", {}},       // Temporary, subsequent PR will remove this.
      {"training_dropout_default_mask", "result differs", {}},  // Temporary, subsequent PR will remove this.
      {"training_dropout_mask", "result differs", {}},          // Temporary, subsequent PR will remove this.
      {"batchnorm_epsilon_training_mode", "training only", {}},
      {"batchnorm_example_training_mode", "training only", {}},
      {"bernoulli", "type error", {}},
      {"bernoulli_double", "type error", {}},
      {"bernoulli_double_expanded", "type error", {}},
      {"bernoulli_expanded", "type error", {}},
      {"bernoulli_seed", "type error", {}},
      {"bernoulli_seed_expanded", "type error", {}},
      {"castlike_BFLOAT16_to_FLOAT", "type error", {}},
      {"castlike_BFLOAT16_to_FLOAT_expanded", "type error", {}},
      {"castlike_FLOAT_to_BFLOAT16", "type error", {}},
      {"castlike_FLOAT_to_BFLOAT16_expanded", "type error", {}},
      {"castlike_FLOAT_to_STRING", "type error", {}},
      {"castlike_FLOAT_to_STRING_expanded", "type error", {}},
      {"convtranspose_autopad_same", "Test data has been corrected in ONNX 1.10.", {"opset13", "opset14"}},
      {"gru_batchwise", "type error", {}},
      {"lstm_batchwise", "type error", {}},
      {"optional_get_element", "type error", {}},
      {"optional_get_element_sequence", "type error", {}},
      {"optional_has_element", "type error", {}},
      {"optional_has_element_empty", "type error", {}},
      {"shape_end_1", "type error", {}},
      {"shape_end_negative_1", "type error", {}},
      {"shape_start_1", "type error", {}},
      {"shape_start_1_end_2", "type error", {}},
      {"shape_start_1_end_negative_1", "type error", {}},
      {"shape_start_negative_1", "type error", {}},
      {"simple_rnn_batchwise", "type error", {}},
      {"mod_float_mixed_sign_example", "fmod attribute must be true for floating point types", {}},
      {"col2im_pads", "result mismatch", {"opset18"}},
      {"reduce_l1_empty_set", "unknown version", {}},
      {"reduce_l1_empty_set_expanded", "unknown version", {}},
      {"reduce_l2_empty_set", "unknown version", {}},
      {"reduce_l2_empty_set_expanded", "unknown version", {}},
      {"reduce_log_sum_empty_set", "unknown version", {}},
      {"reduce_log_sum_empty_set_expanded", "unknown version", {}},
      {"reduce_log_sum_exp_empty_set", "unknown version", {}},
      {"reduce_log_sum_exp_empty_set_expanded", "unknown version", {}},
      {"reduce_prod_empty_set", "unknown version", {}},
      {"reduce_sum_empty_set", "unknown version", {}},
      {"reduce_sum_square_empty_set_expanded", "unknown version", {}},
      {"averagepool_3d_dilations_large_count_include_pad_is_1_ceil_mode_is_True", "TODO(titaiwang): enable this in the next ONNX release."},
#ifdef ENABLE_TRAINING_CORE
      {"adagrad", "not a registered function/op", {}},                  // Op not registered.
      {"adagrad_multiple", "not a registered function/op", {}},         // Op not registered.
      {"adam", "not a registered function/op", {}},                     // Op not registered.
      {"adam_multiple", "not a registered function/op", {}},            // Op not registered.
      {"gradient_of_add", "not a registered function/op", {}},          // Op not registered.
      {"gradient_of_add_and_mul", "not a registered function/op", {}},  // Op not registered.
      {"momentum", "not a registered function/op", {}},                 // Op not registered.
      {"momentum_multiple", "not a registered function/op", {}},        // Op not registered.
      {"nesterov_momentum", "not a registered function/op", {}},        // Op not registered.
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_weight_ignore_index_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_log_prob",
       "type error",
       {"opset12"}},
      {"softmax_cross_entropy_mean_weight_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_weight_ignore_index_3d", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_weight_ignore_index_4d_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_weight_ignore_index_4d", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_log_prob",
       "type error",
       {"opset12"}},
      {"softmax_cross_entropy_mean_3d_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_none_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_3d", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_weight_ignore_index_3d_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index_3d_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_none_weights_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_sum_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_weight_ignore_index", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index_3d", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index", "type error", {"opset12"}},
      {"softmax_cross_entropy_sum", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob",
       "type error",
       {"opset12"}},
      {"softmax_cross_entropy_none_weights", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index_4d_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_none", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_weight", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index_4d", "type error", {"opset12"}},
#endif
      {"mask_rcnn_keras", "this model currently has an invalid contrib op version set to 10", {}},
      // ONNX 1.16.0 fix: https://github.com/onnx/onnx/pull/5741
      // ORT pending PR: https://github.com/microsoft/onnxruntime/pull/18377
      {"maxpool_2d_ceil_output_size_reduce_by_one",
       "ONNX 1.16.0 fixed maxpool output size bug and added this test. "
       "Enable when merge: https://github.com/microsoft/onnxruntime/pull/18377",
       {}},
      {"dequantizelinear_blocked", "blocked quantization (onnx 1.16.0) not supported", {}},
      {"quantizelinear_blocked_asymmetric", "blocked quantization (onnx 1.16.0) not supported", {}},
      {"quantizelinear_blocked_symmetric", "blocked quantization (onnx 1.16.0) not supported", {}},
      // See PR that fixes int4 q/dq tests: https://github.com/onnx/onnx/pull/6122
      {"dequantizelinear_int4", "Bug with model input name 'zero_point' not matching node's input name", {}},
      {"dequantizelinear_uint4", "Bug with model input name 'zero_point' not matching node's input name", {}},
      {"quantizelinear_int4", "Bug with model input name 'zero_point' not matching node's input name", {}},
      {"quantizelinear_uint4", "Bug with model input name 'zero_point' not matching node's input name", {}},
      {"qlinearmatmul_2D_int8_float16", "fp16 type ont supported by CPU EP", {}},
      {"qlinearmatmul_2D_int8_float32", "result diff", {}},
      {"qlinearmatmul_2D_uint8_float16", "fp16 type ont supported by CPU EP", {}},
      {"qlinearmatmul_3D_int8_float16", "fp16 type ont supported by CPU EP", {}},
      {"qlinearmatmul_3D_int8_float32", "result diff", {}},
      {"qlinearmatmul_3D_uint8_float16", "fp16 type ont supported by CPU EP", {}}});

  // Some EPs may fail to pass some specific testcases.
  // For example TenosrRT EP may fail on FLOAT16 related testcases if GPU doesn't support float16.
  // Instead of list all these testcases, we can use following keyword set to filter out testcases wchich contain
  // specific keyword.
  // std::set<std::string> broken_tests_keyword_set = {};

  if (provider_name == "cuda") {
#ifdef ENABLE_TRAINING_CORE
    // cudnn frontend exception in orttraining-linux-gpu-ci-pipeline.
    broken_tests->insert({"keras_lotus_resnet3D", "Temporarily disabled pending investigation", {}});
#endif
#ifdef _WIN32
    broken_tests->insert({"LSTM_Seq_lens_unpacked", "this test fails with new image since Aug 25."});
    broken_tests->insert({"bidaf", "this test fails with new image since Aug 25."});
    broken_tests->insert({"Candy", "Flaky test, need to investigate", {"opset9"}});
#else
    broken_tests->insert({"bidaf", "this test should be recovered when multi-gpu pipeline deprecates NV12", {"opset9"}});
#endif
  }

  if (provider_name == "nnapi") {
    broken_tests->insert({"scan9_sum", "Error with the extra graph"});
    broken_tests->insert({"scan_sum", "Error with the extra graph"});
    broken_tests->insert({"mvn_expanded", "Failed to find kernel for MemcpyFromHost(1) (node Memcpy_1)"});
    broken_tests->insert({"dynamicquantizelinear_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"dynamicquantizelinear_max_adjusted_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"dynamicquantizelinear_min_adjusted_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"gemm_transposeB", "Temporarily disabled pending investigation"});
    broken_tests->insert({"range_float_type_positive_delta_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"range_int32_type_negative_delta_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"convtranspose_1d", "1d convtranspose not supported yet"});
    broken_tests->insert({"convtranspose_3d", "3d convtranspose not supported yet"});
    broken_tests->insert({"maxpool_2d_uint8", "result mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NC_expanded", "shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_expanded", "shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_mean_expanded", "shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_sum_expanded", "shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_expanded", "shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_mean_expanded", "shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_expanded", "shape mismatch"});
    // Disable based on George Wu's recommendation.
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_ignore_index_expanded",
         "shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_iinput_shape_is_NCd1_weight_ignore_index", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_iinput_shape_is_NCd1_weight_ignore_index_expanded", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NC", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1_expanded", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1_ignore_index", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1_ignore_index_expanded", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1_mean_weight_negative_ignore_index", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1_mean_weight_negative_ignore_index_expanded",
                          "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1_weight", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1_weight_expanded", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_no_weight_reduction_mean_ignore_index", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_no_weight_reduction_mean_ignore_index_expanded",
         "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_mean", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_sum", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_mean", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_ignore_index",
                          "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index",
                          "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded",
         "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_expanded",
                          "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_expanded", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_log_prob", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_log_prob_expanded",
         "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded",
                          "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob",
                          "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob_expanded",
         "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_expanded", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_log_prob_expanded",
                          "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_3d", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_3d_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_3d_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_3d_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_3d", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_3d_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_3d_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_3d_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_4d", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_4d_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_4d_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_4d_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_3d", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_3d_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_3d_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_3d_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_4d", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_4d_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_4d_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_4d_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none_weights", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none_weights_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none_weights_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none_weights_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_sum", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_sum_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_sum_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_sum_log_prob_expanded", "Shape mismatch"});
  }

  if (provider_name == "tensorrt") {
    broken_tests->insert({"convtranspose_with_kernel", "It causes segmentation fault"});
    broken_tests->insert({"convtranspose_pad", "It causes segmentation fault"});
    broken_tests->insert({"convtranspose_kernel_shape", "It causes segmentation fault"});
    broken_tests->insert({"dynamicquantizelinear_expanded", "It causes segmentation fault"});
    broken_tests->insert({"dynamicquantizelinear_min_adjusted_expanded", "It causes segmentation fault"});
    broken_tests->insert({"dynamicquantizelinear_max_adjusted_expanded", "It causes segmentation fault"});

    broken_tests->insert({"basic_conv_with_padding",
                          "Cannot set more than one input unless network has Q/DQ layers. TensorRT EP could not build "
                          "engine for fused node"});
    broken_tests->insert({"basic_conv_without_padding",
                          "Cannot set more than one input unless network has Q/DQ layers. TensorRT EP could not build "
                          "engine for fused node"});
    broken_tests->insert({"conv_with_strides_no_padding",
                          "Cannot set more than one input unless network has Q/DQ layers. TensorRT EP could not build "
                          "engine for fused node"});

    broken_tests->insert({"conv_with_autopad_same",
                          "Internal Error (node_of_y: Cannot set more than one input unless network has Q/DQ layers.)"});

    // unsupported tests since opset16
    broken_tests->insert({"sequence_map_add_2_sequences", "not supported by TensorRT EP"});
    broken_tests->insert({"sequence_map_extract_shapes", "not supported by TensorRT EP."});
    broken_tests->insert({"sequence_map_add_1_sequence_1_tensor", "not supported by TensorRT EP."});
    broken_tests->insert({"sequence_map_identity_1_sequence", "not supported by TensorRT EP."});
    broken_tests->insert({"sequence_map_identity_2_sequences", "not supported by TensorRT EP."});
    broken_tests->insert({"sequence_map_identity_1_sequence_1_tensor", "not supported by TensorRT EP."});
    broken_tests->insert({"leakyrelu_expanded", "not supported by TensorRT EP."});
    broken_tests->insert({"leakyrelu_default_expanded", "not supported by TensorRT EP."});
    broken_tests->insert({"leakyrelu_example_expanded", "not supported by TensorRT EP."});
    broken_tests->insert({"prelu_broadcast_expanded", "not supported by TensorRT EP."});
    broken_tests->insert({"prelu_example_expanded", "not supported by TensorRT EP."});
  }

  if (provider_name == "dml") {
    broken_tests->insert({"tinyyolov3", "The parameter is incorrect"});
    broken_tests->insert({"PixelShuffle", "Test requires 6D Reshape, which isn't supported by DirectML"});
    broken_tests->insert({"operator_permute2", "Test requires 6D Transpose, which isn't supported by DirectML"});
    broken_tests->insert({"resize_downsample_linear",
                          "ORT 0.4 uses asymmetric but will conform to half_pixel in the next ONNX version."});
    broken_tests->insert(
        {"resize_upsample_linear", "ORT 0.4 uses asymmetric but will conform to half_pixel in the next ONNX version."});
    broken_tests->insert(
        {"resize_upsample_linear", "ORT 0.4 uses asymmetric but will conform to half_pixel in the next ONNX version."});

    // These tests are temporarily disabled pending investigation
    broken_tests->insert({"dynamicquantizelinear_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"dynamicquantizelinear_max_adjusted_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"dynamicquantizelinear_min_adjusted_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"mxnet_arcface", "Temporarily disabled pending investigation"});
    broken_tests->insert({"yolov3", "Temporarily disabled pending investigation"});
    broken_tests->insert({"tf_inception_v2", "Temporarily disabled pending investigation"});
    broken_tests->insert({"fp16_inception_v1", "Temporarily disabled pending investigation"});
    broken_tests->insert({"candy", "Temporarily disabled pending investigation"});
    broken_tests->insert({"BERT_Squad", "Temporarily disabled pending investigation"});
    broken_tests->insert({"LSTM_Seq_lens_unpacked", "The parameter is incorrect"});
    broken_tests->insert({"mlperf_ssd_resnet34_1200", "The parameter is incorrect"});

    broken_tests->insert({"resize_downsample_scales_linear",
                          "DML uses half_pixel and this test assumed \"asymmetric\" but does not include \"mode\""});
    broken_tests->insert({"resize_downsample_sizes_linear_pytorch_half_pixel",
                          "DML does not support downsampling by such a large factor - skips input pixels"});
    broken_tests->insert({"resize_downsample_sizes_nearest",
                          "DML uses pixel centers for nearest, rounding 1 value off for the middle column"});
    broken_tests->insert({"resize_upsample_sizes_nearest",
                          "DML uses pixel centers for nearest, which makes more sense (the 3rd row mismatches)"});
    broken_tests->insert({"unsqueeze_three_axes", "DML does not support 6D tensors"});
    broken_tests->insert({"unsqueeze_unsorted_axes", "DMLdoes not support 6D tensors"});

    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded",
         "DML does not support 5D+ tensors"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight", "DML does not support 5D+ tensors"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob_expanded",
         "DML does not support 5D+ tensors"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight", "DML does not support 5D+ tensors"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded", "DML does not support 5D+ tensors"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob", "DML does not support 5D+ tensors"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob_expanded",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight", "DML does not support 5D+ tensors"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob_expanded",
                          "DML does not support 5D+ tensors"});
  }

  if (provider_name == "qnn") {
    broken_tests->insert({"gemm_default_no_bias", "result differs"});
    broken_tests->insert({"resize_downsample_scales_linear", "result differs"});
    broken_tests->insert({"resize_downsample_scales_linear_antialias", "result differs"});
    broken_tests->insert({"resize_downsample_sizes_linear_antialias", "result differs"});
    broken_tests->insert({"sce_NCd1_mean_weight_negative_ii", "result differs"});
    broken_tests->insert({"sce_NCd1_mean_weight_negative_ii_expanded", "result differs"});
    broken_tests->insert({"sce_NCd1_mean_weight_negative_ii_log_prob", "result differs"});
    broken_tests->insert({"sce_NCd1_mean_weight_negative_ii_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean", "result differs"});
    broken_tests->insert({"sce_mean_3d", "result differs"});
    broken_tests->insert({"sce_mean_3d_expanded", "result differs"});
    broken_tests->insert({"sce_mean_3d_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_3d_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_expanded", "result differs"});
    broken_tests->insert({"sce_mean_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_3d", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_3d_expanded", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_3d_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_3d_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_4d", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_4d_expanded", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_4d_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_4d_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_expanded", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight", "result differs"});
    broken_tests->insert({"sce_mean_weight_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_3d", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_3d_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_3d_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_3d_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_4d", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_4d_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_4d_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_4d_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_weight_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_none", "result differs"});
    broken_tests->insert({"sce_none_expanded", "result differs"});
    broken_tests->insert({"sce_none_log_prob", "result differs"});
    broken_tests->insert({"sce_none_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_sum", "result differs"});
    broken_tests->insert({"sce_sum_expanded", "result differs"});
    broken_tests->insert({"sce_sum_log_prob", "result differs"});
    broken_tests->insert({"sce_sum_log_prob_expanded", "result differs"});
    broken_tests->insert({"gridsample_reflection_padding", "result differs"});
    broken_tests->insert({"gridsample_volumetric_nearest_align_corners_0", "unknown version"});
    broken_tests->insert({"gridsample_volumetric_nearest_align_corners_1", "unknown version"});
    broken_tests->insert({"spacetodepth", "result differs"});
    broken_tests->insert({"reduce_sum_square_empty_set_expanded", "unknown version"});
    // Fails with QNN SDK 2.17.0:
    // expected 7.70947 (40f6b3f3), got 7.84096 (40fae920), diff: 0.131491, tol=0.00870947 idx=419. 100 of 1715 differ
    broken_tests->insert({"facedetection_op8_qdq", "result differs"});

#if defined(_WIN32) && defined(_M_AMD64)
    // Fails with QNN SDK 2.17.0 on Windows x64:
    // expected 13.5 (41580000), got 0 (0), diff: 13.5, tol=0.0145 idx=3. 3 of 4 differ
    broken_tests->insert({"averagepool_2d_ceil", "result differs"});
#endif
    // These next 3 Resize tests fail on CPU backend with QNN SDK 2.22.0 due to inaccuracy.
    // output=Y:expected 1 (3f800000), got 3 (40400000), diff: 2, tol=0.002 idx=24. 8 of 56 differ
    broken_tests->insert({"resize_upsample_sizes_nearest", "result differs"});
    broken_tests->insert({"resize_upsample_sizes_nearest_axes_2_3", "result differs"});
    broken_tests->insert({"resize_upsample_sizes_nearest_axes_3_2", "result differs"});
    broken_tests->insert({"resize_upsample_sizes_nearest_not_larger",
                          "output=Y:expected 1 (3f800000), got 4 (40800000), diff: 3, tol=0.002 idx=24. 13 of 49 differ. CPU test passed."});
    broken_tests->insert({"convtranspose_group_2", "Segmentation fault (core dumped). CPU test passed."});
    broken_tests->insert({"convtranspose_group_2_image_3", "Segmentation fault (core dumped). CPU test passed."});
    // Fails with QNN 2.31 on Windows x64 for CPU
    broken_tests->insert({"gelu_tanh_2", "y:expected -0.0131778 (bc57e7d5), got -0.0136333 (bc5f5e38), diff: 0.000455472, tol=2.31778e-05."});
    broken_tests->insert({"convtranspose_pad", "Access violation 0xc000005 from call graphAddNode."});
    broken_tests->insert({"convtranspose_pads", "Access violation 0xc000005 from call graphAddNode."});
    broken_tests->insert({"convtranspose_output_shape", "Access violation 0xc000005 from call graphAddNode."});
    broken_tests->insert({"convtranspose_kernel_shape", "Access violation 0xc000005 from call graphAddNode."});
    broken_tests->insert({"convtranspose_1d", "Access violation 0xc000005 from call graphAddNode."});
    broken_tests->insert({"convtranspose", "Access violation 0xc000005 from call graphAddNode."});
    broken_tests->insert({"averagepool_2d_ceil", "result differs. expected 13.5 (41580000), got 0 (0)"});
  }

#ifdef DISABLE_CONTRIB_OPS
  broken_tests->insert({"coreml_SqueezeNet_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Permute_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_ReLU_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Padding-Upsampling-Normalizer_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"tiny_yolov2", "This model uses contrib ops."});
  broken_tests->insert({"fp16_tiny_yolov2", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Pooling_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Padding_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Normalizer_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_linear_sklearn_load_breast_cancer", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_linear_ImageNet_small", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_linear_ImageNet_large", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_linear_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_leakyrelu_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_hard_sigmoid_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_elu_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Dense_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Conv2D_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"coreml_VGG16_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"coreml_Resnet50_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"coreml_Inceptionv3_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"coreml_FNS-Candy_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"coreml_AgeNet_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_thresholdedrelu_ImageNet_large", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_thresholdedrelu_ImageNet_small", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_thresholdedrelu_sklearn_load_breast_cancer", "This model uses contrib ops."});
  broken_tests->insert({"thresholdedrelu", "This model uses contrib ops."});
  broken_tests->insert({"thresholdedrelu_default", "This model uses contrib ops."});
  broken_tests->insert({"dynamic_slice_default_axes", "This model uses contrib ops."});
  broken_tests->insert({"thresholdedrelu_example", "This model uses contrib ops."});
  broken_tests->insert({"dynamic_slice_neg failed", "This model uses contrib ops."});
  broken_tests->insert({"dynamic_slice_start_out_of_bounds", "This model uses contrib ops."});
  broken_tests->insert({"dynamic_slice", "This model uses contrib ops."});
  broken_tests->insert({"dynamic_slice_end_out_of_bounds", "This model uses contrib ops."});
  broken_tests->insert({"dynamic_slice_neg", "This model uses contrib ops."});
  broken_tests->insert({"mvn", "This model uses contrib ops.", {"onnx130"}});
  broken_tests->insert({"cdist_float32_euclidean_1000_2000_1", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float32_euclidean_1000_2000_500", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float32_euclidean_1_1_1", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float32_sqeuclidean_1000_2000_1", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float32_sqeuclidean_1000_2000_500", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float32_sqeuclidean_1_1_1", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float64_euclidean_1000_2000_1", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float64_euclidean_1000_2000_500", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float64_euclidean_1_1_1", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float64_sqeuclidean_1000_2000_1", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float64_sqeuclidean_1000_2000_500", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float64_sqeuclidean_1_1_1", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Average_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"bidaf", "This model uses contrib ops."});
  broken_tests->insert({"fp16_test_tiny_yolov2", "This model uses contrib ops."});
  broken_tests->insert({"fp16_coreml_FNS-Candy", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Repeat_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_BiDirectional_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"fp16_coreml_LinearRegression_NYCTaxi", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Average_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_GRU_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_SimpleRNN_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Dot_imageNet", "This model uses contrib ops."});
#endif
  return broken_tests;
}

// Some EPs may fail to pass some specific testcases.
// For example TenosrRT EP may fail on FLOAT16 related testcases if GPU doesn't support float16.
// Instead of list all these testcases, we can use following keyword set to filter out testcases wchich contain
// specific keyword.
std::unique_ptr<std::set<std::string>> GetBrokenTestsKeyWordSet(const std::string& provider_name) {
  auto broken_tests_keyword_set = std::make_unique<std::set<std::string>>();
  if (provider_name == "tensorrt") {
    broken_tests_keyword_set->insert({"scatternd_add"});
    broken_tests_keyword_set->insert({"scatternd_multiply"});
    broken_tests_keyword_set->insert({"scatter_elements_with_duplicate_indices"});

    // sce op is not supported
    broken_tests_keyword_set->insert({"sce"});

    // TensorRT EP CI uses Nvidia Tesla M60 which doesn't support fp16.
    broken_tests_keyword_set->insert({"FLOAT16"});
  }
  return broken_tests_keyword_set;
}
