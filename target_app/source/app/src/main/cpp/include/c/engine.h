// Copyright 2025 The ODML Authors.
  
// Licensed under the Apache License, Version 2.0 (the "License");
                                                                   
                                          
  
                                                  
  
                                                                      
                                                                    
                                                                           
                                                                      
                                 

#ifndef THIRD_PARTY_ODML_LITERT_LM_C_ENGINE_H_
#define THIRD_PARTY_ODML_LITERT_LM_C_ENGINE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__APPLE__)
#include "api_export.h"               
#include "error_reporter.h"           
#else
#include "c/api_export.h"
#include "c/error_reporter.h"
#endif

                                                            
  
                                                                           
                       
                                                                        
                                                                            
                                                                  
                               
                                                          
  
                                                                        
                                                                             
                                                                              
                                                
  
                                                                     
                                    
  
                          
#define LITERT_LM_C_API_VERSION_MAJOR 1
#define LITERT_LM_C_API_VERSION_MINOR 0
#define LITERT_LM_C_API_VERSION_PATCH 0

#ifdef __cplusplus
extern "C" {
#endif

                                                                  
                                                                    
  
                                                                              
                                                                     
                                                                      
  
                             
                                
                                                           
                                       
                                                  
                                                 
                                                                               
                                                                              
      
  
                                                                        
                                                                               
                                                                               
                                                        
  
                                                                              
                        
                                                                              
                        
                                                                              
                        
                                                                             
                                                      
  
                          
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_get_api_version(int* out_major, int* out_minor,
                                             int* out_patch);

                                           
  
                          
typedef struct LiteRtLmEngine LiteRtLmEngine;

                                            
  
                          
typedef struct LiteRtLmSession LiteRtLmSession;

                                              
  
                          
typedef struct LiteRtLmResponses LiteRtLmResponses;

                                                    
  
                          
typedef struct LiteRtLmEngineSettings LiteRtLmEngineSettings;

                                                   
  
                          
typedef struct LiteRtLmBenchmarkInfo LiteRtLmBenchmarkInfo;

                                                              
  
                          
typedef struct LiteRtLmRepetitionPenaltyConfig LiteRtLmRepetitionPenaltyConfig;

                                                           
  
                          
typedef struct LiteRtLmNoRepeatNgramConfig LiteRtLmNoRepeatNgramConfig;

                                                           
  
                          
typedef struct LiteRtLmSuppressTokensConfig LiteRtLmSuppressTokensConfig;

                                          
                                                           
  
                          
typedef struct LiteRtLmDetokenizeResult LiteRtLmDetokenizeResult;

                                        
                                                         
  
                          
typedef struct LiteRtLmTokenizeResult LiteRtLmTokenizeResult;

                                       
  
                          
typedef enum {
  kLiteRtLmTokenUnionTypeString = 0,
  kLiteRtLmTokenUnionTypeIds = 1,
} LiteRtLmTokenUnionType;

                                            
                                                                               
                         
                                                     
  
                          
typedef struct LiteRtLmTokenUnion LiteRtLmTokenUnion;

                                             
                                                                       
              
                                                      
  
                          
typedef struct LiteRtLmTokenUnions LiteRtLmTokenUnions;

                                           
                                                    
  
                          
typedef struct LiteRtLmInputData LiteRtLmInputData;

                                               
  
                          
typedef struct LiteRtLmSessionConfig LiteRtLmSessionConfig;

                                  
  
                          
typedef enum {
                                  
  kLiteRtLmSamplerTypeUnspecified = 0,
                                                   
  kLiteRtLmSamplerTypeTopK = 1,
                                                                         
                                                                     
  kLiteRtLmSamplerTypeTopP = 2,
                                                      
  kLiteRtLmSamplerTypeGreedy = 3,
} LiteRtLmSamplerType;

                                                   
                                                        
  
                          
typedef struct LiteRtLmSamplerParams LiteRtLmSamplerParams;

                                                                     
  
                                       
                                                                              
                                                                           
             
                                                                         
                                                                      
                                                                            
  
                          
                                                                               
                      
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_sampler_params_create(
    LiteRtLmSamplerType type, LiteRtLmSamplerParams** out_params);

                                         
  
                                           
  
                          
LITERT_LM_C_API_EXPORT
void litert_lm_sampler_params_delete(LiteRtLmSamplerParams* params);

                        
  
                                                  
                                
                                                                             
                      
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_sampler_params_set_top_k(
    LiteRtLmSamplerParams* params, int32_t top_k);

                        
  
                                                  
                                
                                                                             
                      
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_sampler_params_set_top_p(
    LiteRtLmSamplerParams* params, float top_p);

                        
  
                                                  
                                               
                                                                             
                      
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_sampler_params_set_temperature(
    LiteRtLmSamplerParams* params, float temperature);

                 
  
                                                  
                               
                                                                             
                      
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_sampler_params_set_seed(
    LiteRtLmSamplerParams* params, int32_t seed);

                                      
  
                                                                          
                                                                           
             
                                                                         
                                                                      
                          
  
                          
                                                                          
                      
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_config_create(
    LiteRtLmSessionConfig** out_config);

                                                                             
                                                                               
                                   
                                      
                                                                               
                    
                                                                             
                      
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_config_set_max_output_tokens(
    LiteRtLmSessionConfig* config, int max_output_tokens);

                                                          
                                      
                                                                 
                                                                             
                      
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_config_set_apply_prompt_template(
    LiteRtLmSessionConfig* config, bool apply_prompt_template);

                                                                
                                      
                                                                             
                                                                           
                                                                             
                                                                  
                                                                         
                                                                          
                                                                                
                                                                                
              
                                                                             
                      
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_config_set_enable_speculative_decoding(
    LiteRtLmSessionConfig* config, bool enable_speculative_decoding);

                                                       
                                      
                                                       
                                                                             
                                          
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_config_set_sampler_params(
    LiteRtLmSessionConfig* config, const LiteRtLmSamplerParams* sampler_params);

                                       
                                       
  
                          
LITERT_LM_C_API_EXPORT
void litert_lm_session_config_delete(LiteRtLmSessionConfig* config);

                                          
                                      
                                                           
                                                                         
                                                                               
                                                                              
                                              
  
                          
                                                                             
                                              
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_config_set_lora_path(
    LiteRtLmSessionConfig* config, const char* lora_path);

                                                
                                      
                                                                  
                                                                         
                                                                               
                                                                              
                                                          
  
                          
                                                                             
                                              
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_config_set_audio_lora_path(
    LiteRtLmSessionConfig* config, const char* audio_lora_path);

                                                                    
                                                           
                                                                              
                        
  
                                                                              
                                       
                                                   
                                                                        
  
                                                                          
                                                                              
                     
                                                                         
                                                                      
                          
  
                          
                                                                          
                      
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_repetition_penalty_config_create(
    LiteRtLmRepetitionPenaltyConfig** out_config);

                                                  
                                       
  
                          
LITERT_LM_C_API_EXPORT
void litert_lm_repetition_penalty_config_delete(
    LiteRtLmRepetitionPenaltyConfig* config);

                                                                                
                                      
                                                                                
                                                                               
                                                                                
                                                                             
                                                                         
                                    
                                                                             
                      
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_repetition_penalty_config_set_repetition_penalty(
    LiteRtLmRepetitionPenaltyConfig* config, float repetition_penalty);

                                                                           
                                      
                                                                           
                                                                        
                                                                                
                                           
                                                                             
                      
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_repetition_penalty_config_set_presence_penalty(
    LiteRtLmRepetitionPenaltyConfig* config, float presence_penalty);

                                                                            
                                      
                                                                            
                                                                                
                                                                         
                                                                            
                                                                             
                      
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_repetition_penalty_config_set_frequency_penalty(
    LiteRtLmRepetitionPenaltyConfig* config, float frequency_penalty);

                                                          
                                      
                                                                               
                                                                          
                                                                            
                                                                            
                                                                             
                      
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_repetition_penalty_config_set_window_size(
    LiteRtLmRepetitionPenaltyConfig* config, int window_size);

                                                                 
                                                                              
                        
  
                                                                                
                                                                              
                                                                              
                                                                           
        
  
                                                                          
                                                                           
                     
                                                                         
                                                                      
                          
  
                          
                                                                          
                      
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_no_repeat_ngram_config_create(
    LiteRtLmNoRepeatNgramConfig** out_config);

                                               
                                       
  
                          
LITERT_LM_C_API_EXPORT
void litert_lm_no_repeat_ngram_config_delete(
    LiteRtLmNoRepeatNgramConfig* config);

                                                                
                                      
                                                                               
                                                                              
                                                                         
                                                                              
                                                                              
                                               
                                                                             
                      
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_no_repeat_ngram_config_set_no_repeat_ngram_size(
    LiteRtLmNoRepeatNgramConfig* config, int no_repeat_ngram_size);

                                                       
                                      
                                                                               
                                                                            
                                                                      
                                                                         
                                                                              
                                                                             
                  
                                                                             
                      
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_no_repeat_ngram_config_set_window_size(
    LiteRtLmNoRepeatNgramConfig* config, int window_size);

                                                                               
                                                                    
  
                                                                              
                                                                             
                                                                            
         
  
                                                                          
                                                                           
                     
                                                                         
                                                                      
                          
  
                          
                                                                          
                      
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_suppress_tokens_config_create(
    LiteRtLmSuppressTokensConfig** out_config);

                                               
                                       
  
                          
LITERT_LM_C_API_EXPORT
void litert_lm_suppress_tokens_config_delete(
    LiteRtLmSuppressTokensConfig* config);

                                                                         
                                      
                                                                             
                                                                              
                                                                         
                                                                                
                                 
                                                                            
                                                                             
                                                                            
                                             
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_suppress_tokens_config_set_suppress_tokens(
    LiteRtLmSuppressTokensConfig* config, const int* suppress_tokens,
    size_t num_tokens);

                                       
  
                          
typedef enum {
  kLiteRtLmLogSeverityVerbose = 0,
  kLiteRtLmLogSeverityDebug = 1,
  kLiteRtLmLogSeverityInfo = 2,
  kLiteRtLmLogSeverityWarning = 3,
  kLiteRtLmLogSeverityError = 4,
  kLiteRtLmLogSeverityFatal = 5,
  kLiteRtLmLogSeveritySilent = 1000,
} LiteRtLmLogSeverity;
                                                        
  
                                            
                                                                             
                                                         
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_set_min_log_level(LiteRtLmLogSeverity level);

                                     
  
                          
typedef enum {
  kLiteRtLmInputDataTypeText,
  kLiteRtLmInputDataTypeImage,
  kLiteRtLmInputDataTypeImageEnd,
  kLiteRtLmInputDataTypeAudio,
  kLiteRtLmInputDataTypeAudioEnd,
} LiteRtLmInputDataType;

                                  
  
                                          
                                                                             
          
                                                                      
                                             
                                             
                                                                              
                                                                           
             
                                                                         
                                                                      
                                                                          
                                                          
  
                          
                                                                              
                          
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_input_data_create(
    LiteRtLmInputDataType type, const void* data, size_t size,
    LiteRtLmInputData** out_input_data);

                                   
  
                                               
  
                          
LITERT_LM_C_API_EXPORT
void litert_lm_input_data_delete(LiteRtLmInputData* input_data);

                                     
  
                                                
                                                              
                                                                           
                                                                         
                                                                              
                                                                            
             
                                                                         
                                                                      
                                                                      
                                                                              
                                           
  
                          
                                                                             
                        
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_create(
    const char* model_path, const char* backend_str,
    const char* vision_backend_str, const char* audio_backend_str,
    LiteRtLmEngineSettings** out_settings);

                                                                           
                                                                         
                                                                      
  
                                              
                                                              
                                                                           
                                                                         
                                                                              
                                                                            
             
                                                                         
                                                                      
                                                                        
                                                                              
                                           
  
                          
                                                                             
                        
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_create_from_raw_file_descriptor(
    int fd, const char* backend_str, const char* vision_backend_str,
    const char* audio_backend_str, LiteRtLmEngineSettings** out_settings);

                                      
  
                                           
  
                          
LITERT_LM_C_API_EXPORT
void litert_lm_engine_settings_delete(LiteRtLmEngineSettings* settings);

                                                    
  
                                       
                                                      
                                                                             
                        
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_max_num_tokens(
    LiteRtLmEngineSettings* settings, int max_num_tokens);

                                                  
  
                                       
                                            
                                                                             
                                                                                
                                   
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_num_threads(
    LiteRtLmEngineSettings* settings, int num_threads);

                                                        
  
                                       
                                            
                                                                             
                                                                            
                                          
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_audio_num_threads(
    LiteRtLmEngineSettings* settings, int num_threads);

                                                                              
                                 
  
                                       
                                                                    
                                                                             
                        
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_parallel_file_section_loading(
    LiteRtLmEngineSettings* settings, bool parallel_file_section_loading);

                                                    
  
                                       
                                                                     
             
                                                                             
                        
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_single_threaded_execution(
    LiteRtLmEngineSettings* settings, bool single_threaded_execution);

                                                    
  
                                                                 
  
                                       
                                                      
                                                                             
                        
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_max_num_images(
    LiteRtLmEngineSettings* settings, int max_num_images);

                                                                     
  
                                                                        
                                                                        
            
  
                                       
                                                                          
                                                                             
                        
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_max_vision_tokens_per_image(
    LiteRtLmEngineSettings* settings, int max_vision_tokens_per_image);

                                           
  
                                       
                                        
                                                                             
                                       
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_cache_dir(
    LiteRtLmEngineSettings* settings, const char* cache_dir);

                                                              
  
                                       
                                                 
                                                                             
                                     
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_litert_dispatch_lib_dir(
    LiteRtLmEngineSettings* settings, const char* lib_dir);

                                                                           
                                                                            
                                                                               
                                                                              
                                                                               
                                                                    
                                                                          
                                                                            
LITERT_LM_C_API_EXPORT
int litert_lm_engine_settings_set_npu_token_id_scrubbing(
    LiteRtLmEngineSettings* settings, bool enabled);


                                       
  
                          
typedef enum {
  kLiteRtLmActivationDataTypeFloat32 = 0,
  kLiteRtLmActivationDataTypeFloat16 = 1,
  kLiteRtLmActivationDataTypeInt16 = 2,
  kLiteRtLmActivationDataTypeInt8 = 3,
} LiteRtLmActivationDataType;

                                 
  
                                       
                                                        
                                                                             
                                                                   
                                      
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_activation_data_type(
    LiteRtLmEngineSettings* settings,
    LiteRtLmActivationDataType activation_data_type);

                                                                              
                       
  
                                       
                                                    
                                                                             
                                                                                
                                   
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_prefill_chunk_size(
    LiteRtLmEngineSettings* settings, int prefill_chunk_size);

                                                                            
  
                                       
                                                   
                                                                             
                                                                                
                                   
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_enable_ynnpack(
    LiteRtLmEngineSettings* settings, bool enable_ynnpack);

                                       
  
                                       
                                                                             
                        
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_enable_benchmark(
    LiteRtLmEngineSettings* settings);

                                                      
  
                                       
                                                          
                                                                             
                        
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_num_prefill_tokens(
    LiteRtLmEngineSettings* settings, int num_prefill_tokens);

                                                     
  
                                       
                                                        
                                                                             
                        
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_num_decode_tokens(
    LiteRtLmEngineSettings* settings, int num_decode_tokens);

                                               
  
                                       
                                                                             
                                                                             
                        
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_enable_speculative_decoding(
    LiteRtLmEngineSettings* settings, bool enable_speculative_decoding);

                                                                
                                                                     
                     
  
                                       
                                                                        
                                                                             
                                                                            
                                                           
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_gpu_decode_steps_per_sync(
    LiteRtLmEngineSettings* settings, int num_decode_steps_per_sync);

                                                               
                                                                              
  
                                       
                                                                     
                                                                             
                                                                            
                                                           
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_gpu_wait_for_weight_uploads(
    LiteRtLmEngineSettings* settings, bool wait_for_weight_uploads);

                                                                
  
                                                                           
                                                                         
                                                                              
                                                
  
                                                                            
                                                                               
                                              
  
                                       
                                                                              
             
                                                                             
                                                                            
                                                           
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_engine_settings_set_use_ringbuffers_local_attention(
    LiteRtLmEngineSettings* settings, bool use_ringbuffers_local_attention);

                                     
  
                                       
                                  
                                                                             
                        
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_lora_rank(
    LiteRtLmEngineSettings* settings, int lora_rank);

                                                
  
                                       
                                                      
                                                     
                                                                         
                                                                      
                                                            
  
                          
                                                                           
                 
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_supported_lora_ranks(
    LiteRtLmEngineSettings* settings, const int* lora_ranks, size_t num_ranks);

                                           
  
                                       
                                        
                                                                             
                                                                            
                                          
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_audio_lora_rank(
    LiteRtLmEngineSettings* settings, int lora_rank);

                                                      
  
                                       
                                                            
                                                     
                                                                         
                                                                      
                                                            
                                                                          
  
                          
                                                                           
                 
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_supported_audio_lora_ranks(
    LiteRtLmEngineSettings* settings, const int* lora_ranks, size_t num_ranks);

                                                     
  
                                                                               
                                                                           
                                                                         
                       
  
                                                                               
                                                                        
                                                              
  
                                       
                                                                           
                                                                             
                                                                             
                                                               
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_settings_set_gpu_enable_metal_residency_set(
    LiteRtLmEngineSettings* settings, bool enable_metal_residency_set);

                                                      
  
                                       
                                                                          
                                                                            
                                                                         
                                                                      
                                                                        
                                                                             
  
                          
                                                                          
                      
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_create(
    const LiteRtLmEngineSettings* settings, LiteRtLmEngine** out_engine);

                         
                         
                                                                        
                                                                           
                          
  
                                                                         
                                                                        
                                                                          
                      

                               
  
                                                                         
                                                                    
  
                                       
  
                          
LITERT_LM_C_API_EXPORT
void litert_lm_engine_delete(LiteRtLmEngine* engine);

                               
  
                                                       
                                                                            
                  
                                                                            
                                                                             
                                                                         
                                                                               
                                                                          
                                
  
                          
                                                                           
                       
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_create_session(
    LiteRtLmEngine* engine, LiteRtLmSessionConfig* config,
    LiteRtLmSession** out_session);

                                
  
                                         
  
                          
LITERT_LM_C_API_EXPORT
void litert_lm_session_delete(LiteRtLmSession* session);

                                                 
  
                                                      
                                                                             
                       
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_cancel_process(LiteRtLmSession* session);

                                                                               
  
                                                     
                                         
                                                                         
                                                                                
                        
  
                                                                           
                 
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_save_checkpoint(LiteRtLmSession* session,
                                                     const char* label);

                                                     
  
                                        
                                                     
                                                                         
                                                                                
                                                                                
                     
  
                                                                           
                 
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_rewind_to_checkpoint(
    LiteRtLmSession* session, const char* label);

                                                 
  
                                        
                                            
                                                                         
                                                                                
                                                                      
  
                                                                           
                 
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_rewind_to_step(LiteRtLmSession* session,
                                                    int step);

                                                                       
                                                                         
                           
  
                                     
                                                                          
           
                                                                  
                                                                         
                                                                                
                                              
  
                          
                                                                           
                 
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_run_prefill(
    LiteRtLmSession* session, const LiteRtLmInputData* const* inputs,
    size_t num_inputs);

                                                                          
                                                                             
                                                                         
                   
  
                                     
                                                                        
                                                                               
                                                                         
                                                                      
                                                                       
  
                          
                                                                          
                         
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_run_decode(
    LiteRtLmSession* session, LiteRtLmResponses** out_responses);

                                                            
  
                                     
                                                               
                                                                     
                                                                              
                            
                                                                        
                                                                               
                                                                         
                                                                      
                                                                               
                                 
  
                          
                                                                          
                         
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_run_text_scoring(
    LiteRtLmSession* session, const char** target_text, size_t num_targets,
    bool store_token_lengths, LiteRtLmResponses** out_responses);

                                           
  
                                                    
                                                                       
             
           
                                                                          
                                                                        
                                                                               
                                                                         
                                                                      
                                                                          
                                                                
  
                          
                                                                          
                         
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_generate_content(
    LiteRtLmSession* session, const LiteRtLmInputData* const* inputs,
    size_t num_inputs, LiteRtLmResponses** out_responses);

                                         
  
                                             
  
                          
LITERT_LM_C_API_EXPORT
void litert_lm_responses_delete(LiteRtLmResponses* responses);

                                             
  
                                                                            
                                                                             
                                                                     
                                                                             
                                                    
  
                                         
                                                                               
                        
                                                                         
                                                                      
                                                 
  
                          
                                                                         
                              
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_responses_get_num_candidates(
    const LiteRtLmResponses* responses, int* out_num_candidates);

                                              
  
                                         
                                          
                                                                              
                                                                              
                
                                                                         
                                                                      
                                                                               
                                                                            
                              
  
                          
                                                                        
                    
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_responses_get_response_text_at(
    const LiteRtLmResponses* responses, int index, const char** out_text);

                                                                    
  
                                         
                                          
                                                                            
                                                      
                                                                         
                                                                      
                                                                         
                                                
  
                          
                                                                          
                         
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_responses_has_score_at(
    const LiteRtLmResponses* responses, int index, bool* out_has_score);

                                      
  
                                         
                                          
                                                                           
                                                                         
                                                                      
                                                                             
                                                                               
                          
  
                          
                                                                         
                     
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_responses_get_score_at(
    const LiteRtLmResponses* responses, int index, float* out_score);

                                                                           
  
                                         
                                          
                                                                             
                                                                   
                                                                         
                                                                      
                                                                             
                                                   
  
                          
                                                                          
                                
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_responses_has_token_length_at(
    const LiteRtLmResponses* responses, int index, bool* out_has_token_length);

                                             
  
                                         
                                          
                                                                             
                
                                                                         
                                                                      
                                                                            
                                                                              
                                          
  
                          
                                                                       
                                     
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_responses_get_token_length_at(
    const LiteRtLmResponses* responses, int index, int* out_token_length);

                                                                         
  
                                         
                                          
                                                                            
                                                                   
                                                                         
                                                                      
                                                                             
                                                   
  
                          
                                                                          
                                
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_responses_has_token_scores_at(
    const LiteRtLmResponses* responses, int index, bool* out_has_token_scores);

                                                                              
  
                                         
                                          
                                                                               
                            
                                                                         
                                                                      
                                                                             
                                                                              
                                              
  
                          
                                                                         
                                
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_responses_get_num_token_scores_at(
    const LiteRtLmResponses* responses, int index, int* out_num_token_scores);

                                             
  
                                         
                                          
                                                                               
                                              
                                                                             
                                                                          
             
                                                                         
                                                                      
                                                                            
                                                                              
                                           
  
                          
                                                                         
                            
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_responses_get_token_scores_at(
    const LiteRtLmResponses* responses, int index,
    const float** out_token_scores);

                                                        
  
                                                             
                                                                              
                                                                               
             
                                                                         
                                                                      
                                                                           
                                           
  
                          
                                                                          
                              
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_get_benchmark_info(
    LiteRtLmSession* session, LiteRtLmBenchmarkInfo** out_benchmark_info);

                                              
  
                                                       
  
                          
LITERT_LM_C_API_EXPORT
void litert_lm_benchmark_info_delete(LiteRtLmBenchmarkInfo* benchmark_info);

                                                  
  
                                                                 
                                                                           
                                               
  
                                                   
                                                                         
                                     
                                                                         
                                                                      
                                               
  
                          
                                                                         
                       
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_benchmark_info_get_time_to_first_token(
    const LiteRtLmBenchmarkInfo* benchmark_info, double* out_seconds);

                                                    
  
                                                   
                                                                           
                                     
                                                                         
                                                                      
                                               
  
                          
                                                                         
                       
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_benchmark_info_get_total_init_time_in_second(
    const LiteRtLmBenchmarkInfo* benchmark_info, double* out_seconds);

                                       
  
                                                   
                                                                             
                        
                                                                         
                                                                      
                                                 
  
                          
                                                                         
                         
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_benchmark_info_get_num_prefill_turns(
    const LiteRtLmBenchmarkInfo* benchmark_info, int* out_num_turns);

                                      
  
                                                   
                                                                            
                        
                                                                         
                                                                      
                                                 
  
                          
                                                                         
                         
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_benchmark_info_get_num_decode_turns(
    const LiteRtLmBenchmarkInfo* benchmark_info, int* out_num_turns);

                                                         
  
                                                   
                                              
                                                                           
                        
                                                                         
                                                                      
                                                                             
                                                      
  
                          
                                                                         
                           
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_benchmark_info_get_prefill_token_count_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index,
    int* out_token_count);

                                                        
  
                                                   
                                             
                                                                          
                        
                                                                         
                                                                      
                                                                             
                                                     
  
                          
                                                                         
                           
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_benchmark_info_get_decode_token_count_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index,
    int* out_token_count);

                                                               
  
                                                   
                                              
                                                                        
                                                                               
                                                                         
                                                                      
                                                      
                                                                                
  
                          
                                                                         
                              
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index,
    double* out_tokens_per_sec);

                                                              
  
                                                   
                                             
                                                                              
                                                                        
                                                                         
                                                                      
                                                      
                                                                               
  
                          
                                                                         
                              
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_benchmark_info_get_decode_tokens_per_sec_at(
    const LiteRtLmBenchmarkInfo* benchmark_info, int index,
    double* out_tokens_per_sec);

                                             
                                                                           
                                                                       
            
  
                          
typedef struct LiteRtLmStreamChunk LiteRtLmStreamChunk;

                                      
  
                                 
                                                                              
                                                                      
                                                                             
                                                           
                                                                               
                                                                     
                                                                     
  
                          
                                                                        
                    
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_stream_chunk_get_text(
    const LiteRtLmStreamChunk* chunk, const char** out_text);

                                                         
  
                                 
                                                                              
                                                         
                                                                         
                                                                              
                               
  
                          
                                                                          
                        
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_stream_chunk_is_final(
    const LiteRtLmStreamChunk* chunk, bool* out_is_final);

                                                             
  
                                 
                                                                           
                                                                             
                                                                         
                                                                            
                                                                             
                                                                      
  
                          
                                                                           
                     
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_stream_chunk_get_error(
    const LiteRtLmStreamChunk* chunk, const char** out_error);

                                    
                                                                         
                                                                             
                                
  
                          
typedef void (*LiteRtLmStreamCallback)(void* callback_data,
                                       const LiteRtLmStreamChunk* chunk);

                                                                          
                                                                             
                                                                         
  
                                     
                                                                    
                                                                         
            
                                                                         
                                                                                
                                                                              
                                                                             
                                   
  
                          
                                                                           
                 
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_run_decode_async(
    LiteRtLmSession* session, LiteRtLmStreamCallback callback,
    void* callback_data);

                                                                         
                                                                             
                                    
  
                                                    
                                                                       
             
           
                                                                          
                                                                    
                                                                         
            
                                                                         
                                                                                
                                                                              
                                                                             
                                   
  
                          
                                                                           
                 
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_session_generate_content_stream(
    LiteRtLmSession* session, const LiteRtLmInputData* const* inputs,
    size_t num_inputs, LiteRtLmStreamCallback callback, void* callback_data);

                                               
  
                                     
                                            
                                                                           
                                                                            
             
                                                                         
                                                                      
                                                                        
                     
  
                          
                                                                          
                      
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_tokenize(
    LiteRtLmEngine* engine, const char* text,
    LiteRtLmTokenizeResult** out_result);

                                        
  
                                                
  
                          
LITERT_LM_C_API_EXPORT
void litert_lm_tokenize_result_delete(LiteRtLmTokenizeResult* result);

                                                
  
                                     
                                                                            
                                        
                                                                              
                                                                             
                                       
                                                                         
                                                                               
                             
  
                          
                                                                         
                      
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_tokenize_result_get_tokens(
    const LiteRtLmTokenizeResult* result, const int** out_tokens);

                                                          
  
                                     
                                                                          
                        
                                                                         
                                                                               
                                 
  
                          
                                                                         
                          
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_tokenize_result_get_num_tokens(
    const LiteRtLmTokenizeResult* result, size_t* out_num_tokens);

                                                      
  
                                     
                                                     
                                                          
                                                                             
                                                                              
             
                                                                         
                                                                      
                                                                          
                     
  
                          
                                                                          
                      
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_detokenize(
    LiteRtLmEngine* engine, const int* tokens, size_t num_tokens,
    LiteRtLmDetokenizeResult** out_result);

                                          
  
                                                  
  
                          
LITERT_LM_C_API_EXPORT
void litert_lm_detokenize_result_delete(LiteRtLmDetokenizeResult* result);

                                               
  
                                       
                                                                         
                                                                               
                            
                                                                         
                                                                               
                           
  
                          
                                                                          
                    
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_detokenize_result_get_string(
    const LiteRtLmDetokenizeResult* result, const char** out_text);

                                    
  
                                                 
  
                          
LITERT_LM_C_API_EXPORT
void litert_lm_token_union_delete(LiteRtLmTokenUnion* token_union);

                                       
  
                                      
                                                                        
                        
                                                                         
                                                                      
                                         
  
                          
                                                                        
                    
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_token_union_get_type(
    const LiteRtLmTokenUnion* token_union, LiteRtLmTokenUnionType* out_type);

                                               
  
                                      
                                                                               
                                                                           
                     
                                                                         
                                                                      
                                                                              
                                       
  
                          
                                                                          
                      
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_token_union_get_string(
    const LiteRtLmTokenUnion* token_union, const char** out_string);

                                            
  
                                      
                                                                            
                                                                          
                                                  
                                                                          
                        
                                                                         
                                                                          
                                                         
                                
  
                          
                                                                           
                 
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_token_union_get_ids(
    const LiteRtLmTokenUnion* token_union, const int** out_tokens,
    size_t* out_num_tokens);

                                            
  
                                                    
  
                          
LITERT_LM_C_API_EXPORT
void litert_lm_token_unions_delete(LiteRtLmTokenUnions* tokens);

                                                        
  
                                         
                                                                             
                        
                                                                         
                                                                               
                                 
  
                          
                                                                         
                          
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_token_unions_get_num_tokens(
    const LiteRtLmTokenUnions* tokens, size_t* out_num_tokens);

                                                                        
  
                                             
                                             
                                                                         
                                                                               
                            
                                                                         
                                                                               
                                                                           
            
  
                          
                                                                          
                     
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_token_unions_get_token_at(
    const LiteRtLmTokenUnions* tokens, size_t index,
    LiteRtLmTokenUnion** out_token);

                                                    
  
                                     
                                                                              
                                                                             
                                                                      
                                                                         
                                                               
                                                                     
                         
  
                          
                                                                          
                     
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_get_start_token(
    LiteRtLmEngine* engine, LiteRtLmTokenUnion** out_token);

                                            
  
                                     
                                                                              
                                                                               
                                                                        
             
                                                                          
                                                               
                                                                     
                          
  
                          
                                                                          
                      
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_engine_get_stop_tokens(
    LiteRtLmEngine* engine, LiteRtLmTokenUnions** out_tokens);

#ifdef __cplusplus
}               
#endif

#endif                                           
