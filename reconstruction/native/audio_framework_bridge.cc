#include "audio_framework_bridge.h"
#include <stdexcept>
#include "absl/types/span.h"
#include "litert/cc/litert_environment.h"
#include "runtime/executor/audio/audio_executor_settings.h"
#include "runtime/executor/audio_litert_compiled_model_executor.h"
#include "support/preprocessor/audio_preprocessor_miniaudio.h"

namespace {
template<class T> T take(absl::StatusOr<T>&& value){
 if(!value.ok())throw std::runtime_error(value.status().ToString());
 return std::move(*value);
}
void check(const absl::Status& status){if(!status.ok())throw std::runtime_error(status.ToString());}

class FrameworkAudioImpl final:public FrameworkAudio {
 litert::Environment env_;
 std::unique_ptr<litert::lm::AudioLiteRtCompiledModelExecutor> executor_;
 public:
 FrameworkAudioImpl(LiteRtEnvironment env,const std::string& model)
  :env_(litert::Environment::WrapCObject(env,litert::OwnHandle::kNo)){
  using namespace litert::lm;
  auto assets=take(ModelAssets::Create(model));
  auto settings=take(AudioExecutorSettings::CreateDefault(assets,0,Backend::NPU,Backend::CPU));
  settings.SetAudioBufferingEnabled(false);
  executor_=take(AudioLiteRtCompiledModelExecutor::Create(std::move(settings),env_));
 }
 FrameworkAudioOutput encode_wav(const std::string& wav,bool debug_logmel)override{
  using namespace litert::support;
                                                                              
  auto frontend=take(AudioPreprocessorMiniAudio::Create(AudioPreprocessorConfig::CreateDefaultGemma4Config(false)));
  auto prepared=take(frontend->Preprocess(InputAudio(std::string(wav))));
  auto mel=take(prepared.GetPreprocessedAudioTensor());
  auto tensor_type=mel->TensorType();if(!tensor_type)throw std::runtime_error(tensor_type.Error().Message());
  auto dims=tensor_type->Layout().Dimensions();
  if(dims.size()!=3||dims[0]!=1||dims[2]!=128)throw std::runtime_error("Unsupported framework audio frontend shape");
  FrameworkAudioOutput out;out.frames=dims[1];
  if(debug_logmel){
   out.logmel.resize(size_t(out.frames)*128);
   auto view=mel->Duplicate();if(!view)throw std::runtime_error(view.Error().Message());
   auto copy=view->Read<float>(absl::MakeSpan(out.logmel));if(!copy)throw std::runtime_error(copy.Error().Message());
  }
  check(executor_->Reset());
  auto encoded=take(executor_->Encode(*mel));
  out.tokens=encoded.GetValidTokens();
  if(out.tokens<=0)throw std::runtime_error("Framework audio encoder produced no valid tokens");
  auto projected=take(encoded.GetProjectedAudioEmbeddingsPtr());
  out.embeddings.resize(size_t(out.tokens)*1536);
  auto view=projected->Duplicate();if(!view)throw std::runtime_error(view.Error().Message());
  auto copied=view->Read<float>(absl::MakeSpan(out.embeddings));
  if(!copied)throw std::runtime_error(copied.Error().Message());
  return out;
 }
};
}
std::unique_ptr<FrameworkAudio> create_framework_audio(LiteRtEnvironment env,const std::string& model){
 return std::make_unique<FrameworkAudioImpl>(env,model);
}
