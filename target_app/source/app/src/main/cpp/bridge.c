#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <jni.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "c/engine.h"
#include "c/error_reporter.h"
typedef struct { LiteRtLmEngine* engine; LiteRtLmSession* session; int gemma3; int qwen; int llama; int ministral; int phi; int vision; int generic; int configured; float temperature; float top_p; int top_k; int max_output_tokens; } Model;
static void* vendor_handle=NULL;
extern int ghost_fix_vendor_bulk_io(void*);
extern void ghost_settings_select_compact_signatures(LiteRtLmEngineSettings*,const char*);
static void fail(JNIEnv* env,const char* fallback){const char* error=litert_lm_get_last_error_message();(*env)->ThrowNew(env,(*env)->FindClass(env,"java/lang/IllegalStateException"),error?error:fallback);}
static LiteRtLmSession* create_session(Model* m){
 LiteRtLmSessionConfig* config=NULL;
 litert_lm_session_config_create(&config);
 if(!config)return NULL;
 litert_lm_session_config_set_max_output_tokens(config,m->configured?m->max_output_tokens:(m->generic?64:512));
 if(m->generic||m->configured){
  LiteRtLmSamplerParams* sampler=NULL;
  if(litert_lm_sampler_params_create(kLiteRtLmSamplerTypeTopP,&sampler)!=kLiteRtLmStatusOk){litert_lm_session_config_delete(config);return NULL;}
  litert_lm_sampler_params_set_top_k(sampler,m->configured&&m->temperature>0?m->top_k:1);litert_lm_sampler_params_set_top_p(sampler,m->configured?m->top_p:1.0f);litert_lm_sampler_params_set_temperature(sampler,m->configured&&m->temperature>0?m->temperature:1.0f);
  litert_lm_session_config_set_sampler_params(config,sampler);litert_lm_sampler_params_delete(sampler);
 }
                                                                               
                                                                                 
 litert_lm_session_config_set_apply_prompt_template(config,false);
 LiteRtLmSession* result=NULL;litert_lm_engine_create_session(m->engine,config,&result);
 litert_lm_session_config_delete(config);
 return result;
}
JNIEXPORT jlong JNICALL Java_com_example_annoymous_NativeModel_openWithBackend(JNIEnv* env,jclass cls,jstring model,jstring cache,jstring libraries,jstring backend,jboolean scrub_token_ids){
 const char* path=(*env)->GetStringUTFChars(env,model,0);const char* dir=(*env)->GetStringUTFChars(env,cache,0);const char* libs=(*env)->GetStringUTFChars(env,libraries,0);
 const char* backend_name=(*env)->GetStringUTFChars(env,backend,0);
 int is_npu=strcmp(backend_name,"npu")==0;
 int gemma3=strstr(path,"Gemma3-1B-IT_q8_")!=NULL;
 int qwen=strstr(path,"qwen3_5_2b")!=NULL;
 int qwen128=qwen&&strstr(path,"_dynvision")!=NULL;
 int llama=strstr(path,"llama3_2_1b")!=NULL;
 int ministral=strstr(path,"ministral3_3b")!=NULL;
 int phi=strstr(path,"phi4_mini")!=NULL;
 int vision=ministral||(qwen&&strstr(path,"_vl_")!=NULL);
 int generic=qwen||llama||ministral||phi;
 int context=qwen&&!vision?256:llama?4096:generic?2048:gemma3?1280:4096;
 char vendor_path[4096];snprintf(vendor_path,sizeof(vendor_path),"%s/libedgetpu_litert.so",libs);
 vendor_handle=dlopen(vendor_path,RTLD_NOW|RTLD_GLOBAL);
 if(!vendor_handle){
  const char* error=dlerror();(*env)->ThrowNew(env,(*env)->FindClass(env,"java/lang/IllegalStateException"),error?error:"Cannot load bundled NPU runtime.");
  (*env)->ReleaseStringUTFChars(env,model,path);(*env)->ReleaseStringUTFChars(env,cache,dir);(*env)->ReleaseStringUTFChars(env,libraries,libs);(*env)->ReleaseStringUTFChars(env,backend,backend_name);return 0;
 }
 ghost_fix_vendor_bulk_io(dlsym(vendor_handle,"thrGetVendorApiVersion"));
 if(!is_npu&&scrub_token_ids==JNI_TRUE){
  (*env)->ReleaseStringUTFChars(env,model,path);(*env)->ReleaseStringUTFChars(env,cache,dir);(*env)->ReleaseStringUTFChars(env,libraries,libs);(*env)->ReleaseStringUTFChars(env,backend,backend_name);
  (*env)->ThrowNew(env,(*env)->FindClass(env,"java/lang/IllegalArgumentException"),"Token ID scrubbing requires the NPU backend.");return 0;
 }
 LiteRtLmEngineSettings* s=NULL;litert_lm_engine_settings_create(path,backend_name,(vision||(!gemma3&&!generic))?"cpu":NULL,(!gemma3&&!generic)?"cpu":NULL,&s);
 (*env)->ReleaseStringUTFChars(env,backend,backend_name);
 (*env)->ReleaseStringUTFChars(env,model,path);
 if(!s){(*env)->ReleaseStringUTFChars(env,cache,dir);(*env)->ReleaseStringUTFChars(env,libraries,libs);fail(env,"Could not create model settings.");return 0;}
 if(is_npu&&litert_lm_engine_settings_set_npu_token_id_scrubbing(s,scrub_token_ids==JNI_TRUE)!=0){(*env)->ReleaseStringUTFChars(env,cache,dir);(*env)->ReleaseStringUTFChars(env,libraries,libs);litert_lm_engine_settings_delete(s);fail(env,"Could not configure NPU token-ID scrubbing.");return 0;}
 if(generic)ghost_settings_select_compact_signatures(s,phi?"decode":ministral?"prefill_32":(qwen&&(vision||qwen128))?"prefill_128":(llama||qwen)?"prefill_8":"prefill_32");
 litert_lm_engine_settings_set_cache_dir(s,dir);litert_lm_engine_settings_set_litert_dispatch_lib_dir(s,libs);litert_lm_engine_settings_set_max_num_tokens(s,context);
 (*env)->ReleaseStringUTFChars(env,cache,dir);(*env)->ReleaseStringUTFChars(env,libraries,libs);
 Model* m=calloc(1,sizeof(Model));if(!m){litert_lm_engine_settings_delete(s);fail(env,"Not enough memory.");return 0;}
 m->gemma3=gemma3;m->qwen=qwen;m->llama=llama;m->ministral=ministral;m->phi=phi;m->vision=vision;m->generic=generic;
 if(generic)setenv("GHOST_LAZY_NPU_LOAD","1",1);
 litert_lm_engine_create(s,&m->engine);litert_lm_engine_settings_delete(s);
 if(generic)unsetenv("GHOST_LAZY_NPU_LOAD");
 if(!m->engine){free(m);fail(env,"Model initialization failed.");return 0;}
 m->session=create_session(m);if(!m->session){fail(env,"Could not create the session.");litert_lm_engine_delete(m->engine);free(m);return 0;}return (jlong)(intptr_t)m;
}
JNIEXPORT jbyteArray JNICALL Java_com_example_annoymous_NativeModel_send(JNIEnv* env,jclass cls,jlong handle,jbyteArray prompt){
 Model* m=(Model*)(intptr_t)handle;if(!m||!m->session){fail(env,"Load the model or start a new session first.");return NULL;}
 jsize n=(*env)->GetArrayLength(env,prompt);
 char* text=calloc((size_t)n+1,1);if(!text){fail(env,"Not enough memory.");return NULL;}
 (*env)->GetByteArrayRegion(env,prompt,0,n,(jbyte*)text);
 if((*env)->ExceptionCheck(env)){free(text);return NULL;}
 LiteRtLmInputData* input=NULL;litert_lm_input_data_create(kLiteRtLmInputDataTypeText,text,(size_t)n,&input);
 memset(text,0,(size_t)n);free(text);
 if(!input){fail(env,"Could not prepare the message.");return NULL;}
 const LiteRtLmInputData* inputs[]={input};
                                                              
 LiteRtLmResponses* response=NULL;litert_lm_session_generate_content(m->session,inputs,1,&response);
 litert_lm_input_data_delete(input);
 if(!response){fail(env,"Generation failed. Start a new session to retry.");return NULL;}
 const char* raw=NULL;litert_lm_responses_get_response_text_at(response,0,&raw);
 jbyteArray out=NULL;
 if(raw){size_t len=strlen(raw);out=(*env)->NewByteArray(env,(jsize)len);if(out)(*env)->SetByteArrayRegion(env,out,0,(jsize)len,(const jbyte*)raw);}else fail(env,"The model returned no text.");
 litert_lm_responses_delete(response);
 return out;
}
JNIEXPORT jintArray JNICALL Java_com_example_annoymous_NativeModel_prefillForCapture(JNIEnv* env,jclass cls,jlong handle,jbyteArray prompt){
 Model* m=(Model*)(intptr_t)handle;
 if(!m||!m->session){fail(env,"Load the model first.");return NULL;}
 jsize n=(*env)->GetArrayLength(env,prompt);
 char* text=calloc((size_t)n+1,1);if(!text){fail(env,"Not enough memory.");return NULL;}
 (*env)->GetByteArrayRegion(env,prompt,0,n,(jbyte*)text);
 if((*env)->ExceptionCheck(env)){free(text);return NULL;}
 LiteRtLmTokenizeResult* tokens=NULL;size_t count=0;const int* ids=NULL;
 LiteRtLmStatusCode status=litert_lm_engine_tokenize(m->engine,text,&tokens);
 if(status==kLiteRtLmStatusOk)status=litert_lm_tokenize_result_get_num_tokens(tokens,&count);
 if(status==kLiteRtLmStatusOk)status=litert_lm_tokenize_result_get_tokens(tokens,&ids);
 if(status!=kLiteRtLmStatusOk){if(tokens)litert_lm_tokenize_result_delete(tokens);memset(text,0,n);free(text);fail(env,"Tokenization failed.");return NULL;}
 LiteRtLmTokenUnion* bos=NULL;const int* bos_ids=NULL;size_t bos_count=0;
 LiteRtLmTokenizeResult* bos_tokens=NULL;
 status=litert_lm_engine_get_start_token(m->engine,&bos);
 if(status==kLiteRtLmStatusOk&&bos){
  LiteRtLmTokenUnionType type;
  status=litert_lm_token_union_get_type(bos,&type);
  if(status==kLiteRtLmStatusOk&&type==kLiteRtLmTokenUnionTypeIds)status=litert_lm_token_union_get_ids(bos,&bos_ids,&bos_count);
  else if(status==kLiteRtLmStatusOk){
   const char* bos_text=NULL;status=litert_lm_token_union_get_string(bos,&bos_text);
   if(status==kLiteRtLmStatusOk)status=litert_lm_engine_tokenize(m->engine,bos_text,&bos_tokens);
   if(status==kLiteRtLmStatusOk)status=litert_lm_tokenize_result_get_num_tokens(bos_tokens,&bos_count);
   if(status==kLiteRtLmStatusOk)status=litert_lm_tokenize_result_get_tokens(bos_tokens,&bos_ids);
  }
 }
 if(status!=kLiteRtLmStatusOk||bos_count>1||count==0||count+bos_count>512){
  if(bos_tokens)litert_lm_tokenize_result_delete(bos_tokens);if(bos)litert_lm_token_union_delete(bos);
  litert_lm_tokenize_result_delete(tokens);memset(text,0,n);free(text);
  (*env)->ThrowNew(env,(*env)->FindClass(env,"java/lang/IllegalArgumentException"),"Capture requires zero or one BOS and 1 to 512 input tokens.");return NULL;
 }
 jintArray result=(*env)->NewIntArray(env,(jsize)(count+bos_count));
 if(result){if(bos_count)(*env)->SetIntArrayRegion(env,result,0,(jsize)bos_count,(const jint*)bos_ids);(*env)->SetIntArrayRegion(env,result,(jsize)bos_count,(jsize)count,(const jint*)ids);}
 if(bos_tokens)litert_lm_tokenize_result_delete(bos_tokens);if(bos)litert_lm_token_union_delete(bos);
 litert_lm_tokenize_result_delete(tokens);
 if(!result||(*env)->ExceptionCheck(env)){memset(text,0,n);free(text);return NULL;}
 LiteRtLmInputData* input=NULL;
 status=litert_lm_input_data_create(kLiteRtLmInputDataTypeText,text,(size_t)n,&input);
 memset(text,0,n);free(text);
 if(status==kLiteRtLmStatusOk){const LiteRtLmInputData* inputs[]={input};status=litert_lm_session_run_prefill(m->session,inputs,1);}
 if(input)litert_lm_input_data_delete(input);
                                                                        
                                                                           
 if(status==kLiteRtLmStatusOk){
  LiteRtLmTokenizeResult* flush_tokens=NULL;size_t flush_count=0;
  status=litert_lm_engine_tokenize(m->engine,"\n",&flush_tokens);
  if(status==kLiteRtLmStatusOk)status=litert_lm_tokenize_result_get_num_tokens(flush_tokens,&flush_count);
  if(flush_tokens)litert_lm_tokenize_result_delete(flush_tokens);
  if(status==kLiteRtLmStatusOk&&flush_count!=1){(*env)->ThrowNew(env,(*env)->FindClass(env,"java/lang/IllegalStateException"),"Flush must be one token.");return NULL;}
  input=NULL;
  if(status==kLiteRtLmStatusOk)status=litert_lm_input_data_create(kLiteRtLmInputDataTypeText,"\n",1,&input);
  if(status==kLiteRtLmStatusOk){const LiteRtLmInputData* inputs[]={input};status=litert_lm_session_run_prefill(m->session,inputs,1);}
  if(input)litert_lm_input_data_delete(input);
 }
 if(status!=kLiteRtLmStatusOk){fail(env,"Prefill capture failed.");return NULL;}
 return result;
}
JNIEXPORT void JNICALL Java_com_example_annoymous_NativeModel_reset(JNIEnv* env,jclass cls,jlong handle){
 Model* m=(Model*)(intptr_t)handle;if(!m)return;
 if(m->session)litert_lm_session_delete(m->session);
 m->session=create_session(m);if(!m->session)fail(env,"Could not start a new session.");
}
JNIEXPORT void JNICALL Java_com_example_annoymous_NativeModel_resetWithGenerationSettings(JNIEnv* env,jclass cls,jlong handle,jfloat temperature,jint top_k,jfloat top_p,jint max_output_tokens){
 Model* m=(Model*)(intptr_t)handle;
 if(!m||!isfinite(temperature)||temperature<0||temperature>2||top_k<1||top_k>1024||!isfinite(top_p)||top_p<=0||top_p>1||max_output_tokens<1||max_output_tokens>4096){
  (*env)->ThrowNew(env,(*env)->FindClass(env,"java/lang/IllegalArgumentException"),"Invalid generation settings or model handle.");return;
 }
 m->configured=1;m->temperature=temperature;m->top_k=top_k;m->top_p=top_p;m->max_output_tokens=max_output_tokens;
 Java_com_example_annoymous_NativeModel_reset(env,cls,handle);
}
JNIEXPORT void JNICALL Java_com_example_annoymous_NativeModel_close(JNIEnv* env,jclass cls,jlong handle){
 Model* m=(Model*)(intptr_t)handle;if(!m)return;
 if(m->session)litert_lm_session_delete(m->session);
 litert_lm_engine_delete(m->engine);free(m);
}

extern LiteRtLmResponses* ghost_session_generate_vl(LiteRtLmEngine*,LiteRtLmSession*,const char*,const char*,int);
extern LiteRtLmResponses* ghost_session_prefill_image(LiteRtLmEngine*,LiteRtLmSession*,const char*,const char*,int);
JNIEXPORT jbyteArray JNICALL Java_com_example_annoymous_NativeModel_prefillAudioForCapture(JNIEnv* env,jclass cls,jlong handle,jbyteArray question,jstring audio){
 Model* m=(Model*)(intptr_t)handle;if(!m||!m->session||m->generic){fail(env,"Load Gemma for audio capture.");return NULL;}
 jsize n=(*env)->GetArrayLength(env,question);char* text=calloc((size_t)n+1,1);if(!text){fail(env,"Not enough memory.");return NULL;}
 (*env)->GetByteArrayRegion(env,question,0,n,(jbyte*)text);if((*env)->ExceptionCheck(env)){free(text);return NULL;}
 const char* path=(*env)->GetStringUTFChars(env,audio,0);
 LiteRtLmResponses* response=ghost_session_prefill_image(m->engine,m->session,text,path,3);
 (*env)->ReleaseStringUTFChars(env,audio,path);memset(text,0,n);free(text);
 if(!response){fail(env,"Audio prefill capture failed.");return NULL;}
 const char* raw=NULL;litert_lm_responses_get_response_text_at(response,0,&raw);jbyteArray out=NULL;
 if(raw){size_t len=strlen(raw);out=(*env)->NewByteArray(env,(jsize)len);if(out)(*env)->SetByteArrayRegion(env,out,0,(jsize)len,(const jbyte*)raw);}
 else fail(env,"Audio prefill returned no metadata.");
 litert_lm_responses_delete(response);return out;
}
JNIEXPORT jbyteArray JNICALL Java_com_example_annoymous_NativeModel_prefillImageForCapture(JNIEnv* env,jclass cls,jlong handle,jbyteArray question,jstring image){
 Model* m=(Model*)(intptr_t)handle;if(!m||!m->session){fail(env,"Load the model first.");return NULL;}
 jsize n=(*env)->GetArrayLength(env,question);char* text=calloc((size_t)n+1,1);if(!text){fail(env,"Not enough memory.");return NULL;}
 (*env)->GetByteArrayRegion(env,question,0,n,(jbyte*)text);if((*env)->ExceptionCheck(env)){free(text);return NULL;}
 const char* path=(*env)->GetStringUTFChars(env,image,0);
 LiteRtLmResponses* response=ghost_session_prefill_image(m->engine,m->session,text,path,m->ministral?1:m->qwen?0:2);
 (*env)->ReleaseStringUTFChars(env,image,path);memset(text,0,n);free(text);
 if(!response){fail(env,"Image prefill capture failed.");return NULL;}
 const char* raw=NULL;litert_lm_responses_get_response_text_at(response,0,&raw);jbyteArray out=NULL;
 if(raw){size_t len=strlen(raw);out=(*env)->NewByteArray(env,(jsize)len);if(out)(*env)->SetByteArrayRegion(env,out,0,(jsize)len,(const jbyte*)raw);}
 else fail(env,"Image prefill returned no metadata.");
 litert_lm_responses_delete(response);return out;
}
extern LiteRtLmResponses* ghost_session_generate_multimodal(LiteRtLmEngine*,LiteRtLmSession*,const char*,const char*,const char*);
JNIEXPORT jbyteArray JNICALL Java_com_example_annoymous_NativeModel_sendMedia(JNIEnv* env,jclass cls,jlong handle,jbyteArray message,jstring image,jstring audio){
 Model* m=(Model*)(intptr_t)handle;if(!m||!m->session){fail(env,"Load the model first.");return NULL;}
 if((m->gemma3||m->generic)&&!m->vision){
  if((*env)->GetStringLength(env,image)||(*env)->GetStringLength(env,audio)){
   (*env)->ThrowNew(env,(*env)->FindClass(env,"java/lang/IllegalArgumentException"),"This model supports text only.");return NULL;
  }
  const char* prefix=m->qwen?"<|im_start|>user\n":m->llama?"<|start_header_id|>user<|end_header_id|>\n\n":m->phi?"<|user|>":"<start_of_turn>user\n";
  const char* suffix=m->qwen?"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n":m->llama?"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n":m->phi?"<|end|><|assistant|>":"<end_of_turn>\n<start_of_turn>model\n";
  jsize n=(*env)->GetArrayLength(env,message),a=(jsize)strlen(prefix),b=(jsize)strlen(suffix);
  jbyteArray turn=(*env)->NewByteArray(env,a+n+b);if(!turn)return NULL;
  (*env)->SetByteArrayRegion(env,turn,0,a,(const jbyte*)prefix);
  jbyte* bytes=(*env)->GetByteArrayElements(env,message,NULL);if(!bytes){(*env)->DeleteLocalRef(env,turn);return NULL;}
  (*env)->SetByteArrayRegion(env,turn,a,n,bytes);(*env)->ReleaseByteArrayElements(env,message,bytes,JNI_ABORT);
  (*env)->SetByteArrayRegion(env,turn,a+n,b,(const jbyte*)suffix);
  jbyteArray result=Java_com_example_annoymous_NativeModel_send(env,cls,handle,turn);
                                                                          
  jbyte* data=(*env)->GetByteArrayElements(env,turn,NULL);if(data){memset(data,0,(size_t)(a+n+b));(*env)->ReleaseByteArrayElements(env,turn,data,0);}
  (*env)->DeleteLocalRef(env,turn);return result;
 }
 jsize n=(*env)->GetArrayLength(env,message);char* text=calloc((size_t)n+1,1);if(!text){fail(env,"Not enough memory.");return NULL;}
 (*env)->GetByteArrayRegion(env,message,0,n,(jbyte*)text);
 if((*env)->ExceptionCheck(env)){free(text);return NULL;}
 const char* image_path=(*env)->GetStringUTFChars(env,image,0);
 const char* audio_path=(*env)->GetStringUTFChars(env,audio,0);
 if(m->vision&&audio_path[0]){
  (*env)->ReleaseStringUTFChars(env,image,image_path);(*env)->ReleaseStringUTFChars(env,audio,audio_path);memset(text,0,(size_t)n);free(text);
  (*env)->ThrowNew(env,(*env)->FindClass(env,"java/lang/IllegalArgumentException"),"This model supports images and text, but not audio.");return NULL;
 }
 LiteRtLmResponses* response=m->vision?ghost_session_generate_vl(m->engine,m->session,text,image_path,m->qwen?0:1):ghost_session_generate_multimodal(m->engine,m->session,text,image_path,audio_path);
 (*env)->ReleaseStringUTFChars(env,image,image_path);(*env)->ReleaseStringUTFChars(env,audio,audio_path);memset(text,0,(size_t)n);free(text);
 if(!response){fail(env,"Multimodal generation failed.");return NULL;}
 const char* raw=NULL;litert_lm_responses_get_response_text_at(response,0,&raw);jbyteArray out=NULL;
 if(raw){size_t len=strlen(raw);out=(*env)->NewByteArray(env,(jsize)len);if(out)(*env)->SetByteArrayRegion(env,out,0,(jsize)len,(const jbyte*)raw);}else fail(env,"The model returned no text.");
 litert_lm_responses_delete(response);return out;
}

extern LiteRtLmResponses* ghost_session_debug(LiteRtLmSession*,const char*);
JNIEXPORT jbyteArray JNICALL Java_com_example_annoymous_NativeModel_inspect(JNIEnv* env,jclass cls,jlong handle,jstring query){
 Model* m=(Model*)(intptr_t)handle;if(!m||!m->session){fail(env,"No active session.");return NULL;}
 const char* request=(*env)->GetStringUTFChars(env,query,0);if(!request)return NULL;
 LiteRtLmResponses* response=ghost_session_debug(m->session,request);
 (*env)->ReleaseStringUTFChars(env,query,request);
 if(!response){fail(env,"Memory inspection failed.");return NULL;}
 const char* raw=NULL;litert_lm_responses_get_response_text_at(response,0,&raw);jbyteArray out=NULL;
 if(raw){size_t len=strlen(raw);out=(*env)->NewByteArray(env,(jsize)len);if(out)(*env)->SetByteArrayRegion(env,out,0,(jsize)len,(const jbyte*)raw);}
 litert_lm_responses_delete(response);return out;
}

                                                                                    
#include <sys/uio.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <errno.h>
JNIEXPORT jbyteArray JNICALL Java_com_example_annoymous_NativeModel_readAddress(JNIEnv* env,jclass cls,jlong address,jint count){
 if(address==0||count<=0||count>4096){(*env)->ThrowNew(env,(*env)->FindClass(env,"java/lang/IllegalArgumentException"),"Invalid address range.");return NULL;}
 unsigned char bytes[4096];struct iovec local={bytes,(size_t)count},remote={(void*)((uintptr_t)address & UINT64_C(0x00ffffffffffffff)),(size_t)count};
 ssize_t n=syscall(__NR_process_vm_readv,getpid(),&local,1,&remote,1,0);
 if(n<0){(*env)->ThrowNew(env,(*env)->FindClass(env,"java/lang/IllegalStateException"),strerror(errno));return NULL;}
 jbyteArray result=(*env)->NewByteArray(env,(jsize)n);if(result)(*env)->SetByteArrayRegion(env,result,0,(jsize)n,(jbyte*)bytes);memset(bytes,0,sizeof(bytes));return result;
}

extern const char* litert_lm_build_litert_revision(void);
JNIEXPORT jstring JNICALL Java_com_example_annoymous_NativeModel_runtimeInfo(JNIEnv* env,jclass cls){
 const char* (*version)(void)=(const char* (*)(void))(vendor_handle?dlsym(vendor_handle,"thrGetVendorApiVersion"):NULL);
 Dl_info info={0};if(version)dladdr((void*)version,&info);
 char result[8192];snprintf(result,sizeof(result),"%s; vendor_api=%s; vendor_path=%s",litert_lm_build_litert_revision(),version?version():"not loaded",info.dli_fname?info.dli_fname:"not loaded");
 return (*env)->NewStringUTF(env,result);
}
