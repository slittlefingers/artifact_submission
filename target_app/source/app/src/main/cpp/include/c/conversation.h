// Copyright 2026 The ODML Authors.
  
// Licensed under the Apache License, Version 2.0 (the "License");
                                                                   
                                          
  
                                                  
  
                                                                      
                                                                    
                                                                           
                                                                      
                                 

#ifndef THIRD_PARTY_ODML_LITERT_LM_C_CONVERSATION_H_
#define THIRD_PARTY_ODML_LITERT_LM_C_CONVERSATION_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__APPLE__)
#include "engine.h"                   
#include "error_reporter.h"           
#else
#include "c/engine.h"
#include "c/error_reporter.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

                                                 
  
                          
typedef struct LiteRtLmConversation LiteRtLmConversation;

                                                               
  
                          
typedef struct LiteRtLmConversationOptionalArgs
    LiteRtLmConversationOptionalArgs;

                                      
  
                          
typedef struct LiteRtLmJsonResponse LiteRtLmJsonResponse;

                                                    
  
                          
typedef struct LiteRtLmConversationConfig LiteRtLmConversationConfig;

                                                    
  
                          
typedef struct LiteRtLmThinkingConfig LiteRtLmThinkingConfig;

                                                              
  
                          
typedef enum {
  kLiteRtLmConstraintTypeNone = 0,
  kLiteRtLmConstraintTypeRegex = 1,
  kLiteRtLmConstraintTypeJsonSchema = 2,
} LiteRtLmConstraintType;

                                              
  
                          
typedef enum {
  kLiteRtLmConstraintProviderTypeLlGuidance = 1,
} LiteRtLmConstraintProviderType;

                                           
  
                                                                          
                                                                             
                
                                                                         
                                                                      
                          
  
                          
                                                                          
                      
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_config_create(
    LiteRtLmConversationConfig** out_config);

                                                        
                                      
                                                   
                                                                             
                                          
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_config_set_session_config(
    LiteRtLmConversationConfig* config,
    const LiteRtLmSessionConfig* session_config);

                                                        
                                      
                                                                
                                                                             
                                               
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_config_set_system_message(
    LiteRtLmConversationConfig* config, const char* system_message_json);

                                               
                                      
                                                                
                                                                             
                                      
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_config_set_tools(
    LiteRtLmConversationConfig* config, const char* tools_json);

                                                          
                                      
                                                                  
                                                                             
                                         
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_config_set_messages(
    LiteRtLmConversationConfig* config, const char* messages_json);

                                                       
                                      
                                                                         
          
                                                                             
                                              
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_config_set_extra_context(
    LiteRtLmConversationConfig* config, const char* extra_context_json);

                                                         
                                      
                                                                              
                                                                
                                                                             
                                           
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_config_set_prompt_template(
    LiteRtLmConversationConfig* config, const char* prompt_template);

                                                                            
                                      
                                                                             
                                                                             
                      
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_conversation_config_set_enable_constrained_decoding(
    LiteRtLmConversationConfig* config, bool enable_constrained_decoding);

                                                                  
                                      
                                                                              
                                                                             
                                                           
                                          
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_config_set_constraint_provider(
    LiteRtLmConversationConfig* config,
    const LiteRtLmConstraintProviderType* provider_type);

                                                            
                                      
                                                                        
           
                                                                             
                      
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_conversation_config_set_filter_channel_content_from_kv_cache(
    LiteRtLmConversationConfig* config,
    bool filter_channel_content_from_kv_cache);

                                           
                                      
                                                               
                                                                    
                                                                             
                      
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_config_set_stream_tool_calls(
    LiteRtLmConversationConfig* config, bool stream_tool_calls,
    const char* channel_name);

                                                                            
       
  
                                                                          
                                                                            
             
                                                                         
                                                                      
                          
  
                          
                                                                          
                      
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_thinking_config_create(
    LiteRtLmThinkingConfig** out_config);

                                        
                                       
  
                          
LITERT_LM_C_API_EXPORT
void litert_lm_thinking_config_delete(LiteRtLmThinkingConfig* config);

                                                         
                                      
                                                      
                                                                             
                      
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_thinking_config_set_enable_thinking(
    LiteRtLmThinkingConfig* config, bool enable_thinking);

                                  
                                      
                                                                              
                     
                                                                             
                      
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_thinking_config_set_thinking_token_budget(
    LiteRtLmThinkingConfig* config, int thinking_token_budget);

                                                         
                                      
                                                                         
                                  
                                                                             
                      
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_config_set_thinking_config(
    LiteRtLmConversationConfig* config,
    const LiteRtLmThinkingConfig* thinking_config);

                                            
                                       
  
                          
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_config_delete(LiteRtLmConversationConfig* config);

                                                  
  
                                                                           
                                      
                                                                           
                                                                         
                                                                      
                                 
  
                          
                                                                         
                                      
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_optional_args_create(
    LiteRtLmConversationOptionalArgs** out_optional_args);

                                                   
                                                     
  
                          
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_optional_args_delete(
    LiteRtLmConversationOptionalArgs* optional_args);

                                                                          
                                       
  
                                                                      
                                                                               
                                                                            
  
                                                                            
          
                                                                               
                                                  
                                                                             
                                                                               
                   
                                                                             
                             
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_conversation_optional_args_set_repetition_penalty_config(
    LiteRtLmConversationOptionalArgs* optional_args,
    const LiteRtLmRepetitionPenaltyConfig* repetition_penalty_config);

                                                                       
                                       
  
                                                                          
                                                                  
                                               
  
                                                                            
          
                                                                         
                                              
                                                                               
                                                                       
                                
                                                                             
                             
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_conversation_optional_args_set_no_repeat_ngram_config(
    LiteRtLmConversationOptionalArgs* optional_args,
    const LiteRtLmNoRepeatNgramConfig* no_repeat_ngram_config);

                                                                       
                                       
  
                                                                             
                                                          
                             
  
                                                                            
          
                                                                         
                                               
                                                                               
                                                                       
                                                                         
                                                                             
                             
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_conversation_optional_args_set_suppress_tokens_config(
    LiteRtLmConversationOptionalArgs* optional_args,
    const LiteRtLmSuppressTokensConfig* suppress_tokens_config);

                                                                   
                                                    
                                                      
                                                                             
                             
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_optional_args_set_visual_token_budget(
    LiteRtLmConversationOptionalArgs* optional_args, int visual_token_budget);

                                                                               
                                                                               
                                   
                                                    
                                                                               
                    
                                                                             
                             
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_optional_args_set_max_output_tokens(
    LiteRtLmConversationOptionalArgs* optional_args, int max_output_tokens);

                                                               
                                                    
                                                                         
                                  
                                                                             
                             
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_optional_args_set_thinking_config(
    LiteRtLmConversationOptionalArgs* optional_args,
    const LiteRtLmThinkingConfig* thinking_config);

                                                          
                                                    
                                                 
                                                                         
                                                                             
                                                                   
                                  
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_optional_args_set_constraint(
    LiteRtLmConversationOptionalArgs* optional_args,
    LiteRtLmConstraintType constraint_type, const char* constraint_string);

                                    
  
                                                            
                                                                            
                  
                                                                               
                                                                             
                
                                                                         
                                                                      
                                                                            
                                                        
  
                          
                                                                       
                                     
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_create(
    LiteRtLmEngine* engine, LiteRtLmConversationConfig* config,
    LiteRtLmConversation** out_conversation);

                                     
  
                                                   
  
                          
LITERT_LM_C_API_EXPORT
void litert_lm_conversation_delete(LiteRtLmConversation* conversation);

                                                                    
  
                                                 
                                                                              
                                                                             
                
                                                                         
                                                                      
                                                                            
                                    
  
                          
                                                                         
                            
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_clone(
    LiteRtLmConversation* conversation,
    LiteRtLmConversation** out_conversation);

                                                                
                           
  
                                               
                                                                      
                                                                            
                                                                   
                                                                           
                                                                          
             
                                                                         
                                                                      
                                                                
                                                                           
           
  
                          
                                                                            
                        
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_send_message(
    LiteRtLmConversation* conversation, const char* message_json,
    const char* extra_context,
    const LiteRtLmConversationOptionalArgs* optional_args,
    LiteRtLmJsonResponse** out_response);

                                             
  
                                           
  
                          
LITERT_LM_C_API_EXPORT
void litert_lm_json_response_delete(LiteRtLmJsonResponse* response);

                                                           
  
                                       
                                                                            
                                                                              
                        
                                                                         
                                                                      
                                      
  
                          
                                                                          
                    
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_json_response_get_string(
    const LiteRtLmJsonResponse* response, const char** out_json);

                                                                     
                                                                             
                                    
  
                                               
                                                                      
                                                                            
                                                                   
                                                                    
                                                                         
            
                                                                         
                                                                      
                                                                             
                                                                          
                                                                             
                                   
  
                          
                                                                             
                                              
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_send_message_stream(
    LiteRtLmConversation* conversation, const char* message_json,
    const char* extra_context,
    const LiteRtLmConversationOptionalArgs* optional_args,
    LiteRtLmStreamCallback callback, void* callback_data);

                                                               
  
                                                                              
                                            
                                                                               
              
  
                                                 
                                                                        
                                                                          
                                                                           
                                                                       
             
                                                                         
                                                                      
                                                                              
                                                               
  
                          
                                                                          
                             
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_render_message_to_string(
    LiteRtLmConversation* conversation, const char* message_json,
    const char** out_text);

                                                               
  
                                                 
                                                                          
                                                                           
                                                                       
             
                                                                         
                                                                      
                                                                              
           
  
                          
                                                                          
                             
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_render_preface_to_string(
    LiteRtLmConversation* conversation, const char** out_text);

                                                                     
  
                                                                    
                                                                             
                            
  
                          
                                                                   
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_cancel_process(
    LiteRtLmConversation* conversation);

                                                                              
                       
  
                                                                              
                                                                     
                                                                               
                                                                             
                                             
  
                                                    
                                                                         
             
  
                          
                                                                       
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_wait_until_done(
    LiteRtLmConversation* conversation);

                                                             
  
                                                                       
                                                                              
                                                                               
             
                                                                         
                                                                      
                                                                              
                                                         
  
                          
                                                                         
                                       
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_get_benchmark_info(
    LiteRtLmConversation* conversation,
    LiteRtLmBenchmarkInfo** out_benchmark_info);

                                                                             
  
                                                 
                                                                             
             
                                                                         
                                                                      
                                                                             
           
  
                          
                                                                              
                                                       
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_conversation_get_token_count(
    LiteRtLmConversation* conversation, int* out_count);

#ifdef __cplusplus
}               
#endif

#endif                                                 
