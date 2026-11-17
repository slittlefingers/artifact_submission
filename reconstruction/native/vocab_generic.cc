#include "litert_support.h"
#include <dlfcn.h>

static void status(const std::string& root,const J& j){savej(root+"/status.tmp",j);require(rename((root+"/status.tmp").c_str(),(root+"/status.json").c_str())==0,"Status rename failed");}
static int collect(const std::vector<std::string>& a){
 const auto& root=a.at(2);
 try{
  J job=J::parse(read(root+"/job.json"));auto meta=job.at("sections");
  void* vendor=dlopen((a[1]+"/libedgetpu_litert.so").c_str(),RTLD_NOW|RTLD_GLOBAL);require(vendor!=nullptr,"Vendor preload failed");
  void* bridge=dlopen((a[1]+"/libannoymous.so").c_str(),RTLD_NOW);require(bridge!=nullptr,"Bridge load failed");
  auto fix=reinterpret_cast<int(*)(void*)>(dlsym(bridge,"ghost_fix_vendor_bulk_io"));require(fix!=nullptr,"Vendor I/O compatibility function missing");fix(dlsym(vendor,"thrGetVendorApiVersion"));
  setenv("GHOST_LAZY_NPU_LOAD","1",1);
  LiteRtEnvOption o{};o.tag=kLiteRtEnvOptionTagDispatchLibraryDir;o.value.type=kLiteRtAnyTypeString;o.value.str_value=a[1].c_str();
  LiteRtEnvironment env;CHECK(LiteRtCreateEnvironment(1,&o,&env));
  int fd=open(a[0].c_str(),O_RDONLY);require(fd>=0,"Model open failed");
  std::string sig=job.at("signature"),emb_sig="prefill_embedder_"+sig.substr(8);
  Section decoder(env,fd,meta.at("tf_lite_prefill_decode"),true,sig),embedder(env,fd,meta.at("tf_lite_embedder"),false,emb_sig);close(fd);
  Signature forward(decoder,sig),embed(embedder,emb_sig);
  int slots=forward.inputs.at("embeddings")->type.layout.dimensions[1];
  auto& vb=*forward.outputs.at("kv_cache_v_0");require(vb.elem==4,"Expected float32 V");
  int heads=vb.type.layout.dimensions[1],dim=vb.type.layout.dimensions[2],capacity=vb.type.layout.dimensions[3];
  std::vector<int> ids;
  if(job.contains("ids"))ids=job.at("ids").get<std::vector<int>>();
  else{ids.resize(job.at("vocabulary").get<int>());std::iota(ids.begin(),ids.end(),0);}
  for(auto& [name,b]:forward.inputs)if(name.rfind("kv_cache_",0)==0)b->put(Bytes(b->count*b->elem,0));
  std::vector<int> positions(slots);std::iota(positions.begin(),positions.end(),0);forward.inputs.at("input_pos")->put(bytes(positions));
  auto& mb=*forward.inputs.at("mask");std::vector<float> mask(mb.count,-0.7f*std::numeric_limits<float>::max());
  require(mb.count==size_t(slots)*capacity,"Unexpected mask shape");
  for(int i=0;i<slots;i++)for(int j=0;j<=i;j++)mask[i*capacity+j]=0;
  mb.put(bytes(mask));
  std::ofstream out(root+"/v.bin",std::ios::binary);require(bool(out),"Cannot open V output");double started=now();
  for(size_t start=0;start<ids.size();start+=slots){
   require(access((root+"/cancel").c_str(),F_OK)!=0,"Cancelled");
   size_t count=std::min(size_t(slots),ids.size()-start);std::vector<int> tokens(slots,0);std::copy_n(ids.begin()+start,count,tokens.begin());
   embed.inputs.at("token_ids")->put(bytes(tokens));embed.run();forward.inputs.at("embeddings")->put(embed.outputs.at("embeddings")->get());forward.run();
   Bytes all=vb.get();std::vector<float> row(heads*dim);
   for(size_t q=0;q<count;q++){
    for(int h=0;h<heads;h++)for(int d=0;d<dim;d++)std::memcpy(&row[h*dim+d],all.data()+((h*dim+d)*capacity+q)*4,4);
    out.write((char*)row.data(),row.size()*4);
   }
   require(bool(out),"V write failed");out.flush();
   status(root,J{{"state","collecting"},{"completed",start+count},{"total",ids.size()},{"width",heads*dim},{"seconds",now()-started}});
  }
  out.close();status(root,J{{"state","complete"},{"completed",ids.size()},{"total",ids.size()},{"width",heads*dim},{"seconds",now()-started}});return 0;
 }catch(const std::exception& e){status(root,J{{"state","failed"},{"error",e.what()}});return 1;}
}
extern "C" JNIEXPORT jint JNICALL Java_com_example_annoymous_ReconstructionVocabDeviceTest_runWorker(JNIEnv* env,jclass,jobjectArray args){
 std::vector<std::string> a;for(jsize i=0;i<env->GetArrayLength(args);i++){auto s=(jstring)env->GetObjectArrayElement(args,i);const char* p=env->GetStringUTFChars(s,nullptr);a.emplace_back(p);env->ReleaseStringUTFChars(s,p);env->DeleteLocalRef(s);}return collect(a);
}
