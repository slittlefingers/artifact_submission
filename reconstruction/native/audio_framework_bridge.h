#pragma once
#include <memory>
#include <string>
#include <vector>
#include "litert/c/litert_environment.h"

struct FrameworkAudioOutput {
 std::vector<float> embeddings;
 std::vector<float> logmel;
 int tokens=0,frames=0;
};

class FrameworkAudio {
 public:
 virtual ~FrameworkAudio()=default;
                                                                              
                                                                            
 virtual FrameworkAudioOutput encode_wav(const std::string& wav,bool debug_logmel)=0;
};
std::unique_ptr<FrameworkAudio> create_framework_audio(LiteRtEnvironment env,const std::string& model);
