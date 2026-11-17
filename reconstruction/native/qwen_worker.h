                                                                              
                                                                                 
#pragma once
#include <dlfcn.h>
#include "qwen_prefill_capture.h"
struct QwenWorker {
 LiteRtEnvironment env=nullptr;
 std::unique_ptr<Section> decoder,embedder;
 std::unique_ptr<Signature> dec,emb,verify;
 int position=0,capacity=2048,slots=32,layer=3;size_t vocab=0;
 Bytes logits;bool rope_zero=false;int rope_delta=0;bool media_prefix=false;
 std::vector<int> prefix_ids;
 std::unique_ptr<QwenPrefillCapture> capture;
 bool exact_prefill=false;
 std::string query_mode="decode";int verify_slots=64;
                                                                           
                                                                              
 Bytes last_exact_row;int last_exact_id=-1,last_exact_position=-1;
 std::vector<std::string> arguments;J settings;
 QwenWorker(const std::vector<std::string>& a,const J& job){
  arguments=a;settings=job;
  layer=job.at("layer");auto meta=job.at("model_sections");
  query_mode=job.value("qwen_query_mode",std::string("decode"));
  require(query_mode=="decode"||query_mode=="exact_prefill"||query_mode=="verify","Invalid Qwen query mode");
  exact_prefill=query_mode=="exact_prefill";
  void* vendor=dlopen((a[1]+"/libedgetpu_litert.so").c_str(),RTLD_NOW|RTLD_GLOBAL);require(vendor,"Vendor preload failed");
  void* bridge=dlopen((a[1]+"/libannoymous.so").c_str(),RTLD_NOW);require(bridge,"Bridge load failed");
  auto fix=reinterpret_cast<int(*)(void*)>(dlsym(bridge,"ghost_fix_vendor_bulk_io"));require(fix,"Vendor I/O fix absent");fix(dlsym(vendor,"thrGetVendorApiVersion"));setenv("GHOST_LAZY_NPU_LOAD","1",1);
  LiteRtEnvOption o{};o.tag=kLiteRtEnvOptionTagDispatchLibraryDir;o.value.type=kLiteRtAnyTypeString;o.value.str_value=a[1].c_str();CHECK(LiteRtCreateEnvironment(1,&o,&env));
  int fd=open(a[0].c_str(),O_RDONLY);require(fd>=0,"Model open failed");
  if(query_mode=="verify"){
   require(meta.at("tf_lite_prefill_decode").at("signatures").contains("verify_64"),"Model has no verify_64 signature");
   decoder=std::make_unique<Section>(env,fd,meta.at("tf_lite_prefill_decode"),true,std::string(),std::vector<std::string>{"decode","verify_64"});
  }else decoder=std::make_unique<Section>(env,fd,meta.at("tf_lite_prefill_decode"),true,"decode");
  embedder=std::make_unique<Section>(env,fd,meta.at("tf_lite_embedder"),false,"decode_embedder");close(fd);
  dec=std::make_unique<Signature>(*decoder,"decode");emb=std::make_unique<Signature>(*embedder,"decode_embedder");
  vocab=dec->outputs.at("logits")->count;capacity=dec->inputs.at("kv_cache_k_"+std::to_string(layer))->type.layout.dimensions[2];
  slots=query_mode=="verify"?128:(job.at("batch_signature")=="prefill_128"?128:32);
  if(query_mode=="verify"){
   verify=std::make_unique<Signature>(*decoder,"verify_64");
   verify_slots=verify->inputs.at("embeddings")->type.layout.dimensions[1];require(verify_slots==64,"Expected verify_64 geometry");
   require(verify->inputs.at("tree")->count==4096,"Expected binary tree matrix");
   for(auto part:{"k","v"}){std::string n="kv_cache_"+std::string(part)+"_"+std::to_string(layer);
    require(verify->outputs.count(n)&&verify->outputs.at(n)->type.element_type==kLiteRtElementTypeFloat32,"Verify selected-layer float32 K/V outputs missing");
    require(verify->outputs.at(n)->layout().at("shape")==dec->outputs.at(n)->layout().at("shape"),"Verify/decode K/V shape mismatch");}
  }
  reset();
 }
 void reset(){for(auto& [n,b]:dec->inputs)if(n.rfind("kv_cache_",0)==0)b->put(Bytes(b->count*b->elem,0));position=0;rope_delta=0;media_prefix=false;logits.clear();prefix_ids.clear();last_exact_row.clear();last_exact_id=-1;last_exact_position=-1;}
 std::pair<Bytes,double> prefill_candidate(int id){
  require(!media_prefix&&prefix_ids.size()==size_t(position)&&position<capacity,"Exact replay exceeds context capacity");
  if(!capture)capture=std::make_unique<QwenPrefillCapture>(arguments,settings);
  auto sequence=prefix_ids;sequence.push_back(id);auto result=capture->collect_sequence(sequence,0);
  require(result.first.size()%sequence.size()==0,"Malformed prefill rows");
  size_t width=result.first.size()/sequence.size();
  last_exact_row=Bytes(result.first.end()-width,result.first.end());
  last_exact_id=id;last_exact_position=position;
  return {last_exact_row,result.second};
 }
 std::pair<Bytes,double> reference_prefill(const std::vector<int>& ids){
  require(settings.value("qwen_recovery",false),"Explicit reference mode required");
  require(!ids.empty()&&ids.size()<=128&&position+128<=capacity,"Invalid sequential reference block");
  if(!capture)capture=std::make_unique<QwenPrefillCapture>(arguments,settings);
                                                                            
                                                                                
  for(auto& [name,b]:capture->prefill->inputs)if(name.rfind("kv_cache_",0)==0)b->put(dec->inputs.at(name)->get());
  return capture->collect(ids,0,position,false);
 }
 Bytes embedding(int id){require(id>=0&&size_t(id)<vocab,"Bad token ID");emb->inputs.at("token_ids")->put(bytes<int32_t>({id}));emb->run();return emb->outputs.at("embeddings")->get();}
 std::pair<Bytes,double> verify_candidates(const std::vector<int>& ids,int& physical_calls){
  require(verify&&query_mode=="verify","Verify was not enabled when the worker was created");
  require(!ids.empty()&&ids.size()<=128,"Verify logical query must contain 1..128 IDs");
  require(position+verify_slots<=capacity,"Verify requires 64 free physical cache slots");
  for(int id:ids)require(id>=0&&size_t(id)<vocab,"Invalid verify candidate");
  Data accepted;for(auto& [n,b]:dec->inputs)if(n.rfind("kv_cache_",0)==0)accepted[n]=b->get();
  std::map<int,Bytes> local_embeddings;for(int id:ids)if(!local_embeddings.count(id))local_embeddings[id]=embedding(id);
  Bytes packed;double elapsed=0;
  std::vector<int32_t> pos(verify_slots),rope(3*verify_slots,rope_zero?0:position+rope_delta);std::iota(pos.begin(),pos.end(),position);
  std::vector<float> tree(verify_slots*verify_slots,0.f),mask(size_t(verify_slots)*capacity,-0.7f*std::numeric_limits<float>::max());
  for(int q=0;q<verify_slots;++q){tree[q*verify_slots+q]=1.f;std::fill(mask.begin()+size_t(q)*capacity,mask.begin()+size_t(q)*capacity+position,0.f);mask[size_t(q)*capacity+position+q]=0.f;}
  verify->inputs.at("input_pos")->put(bytes(pos));verify->inputs.at("rope_pos")->put(bytes(rope));verify->inputs.at("tree")->put(bytes(tree));verify->inputs.at("mask")->put(bytes(mask));
  for(size_t begin=0;begin<ids.size();begin+=verify_slots){
   size_t count=std::min(size_t(verify_slots),ids.size()-begin);
                                                                             
   for(auto& [n,b]:accepted)verify->inputs.at(n)->put(b);
   Bytes embeddings;embeddings.reserve(verify->inputs.at("embeddings")->count*4);
   for(int q=0;q<verify_slots;++q){const auto& e=local_embeddings.at(ids[begin+std::min(size_t(q),count-1)]);embeddings.insert(embeddings.end(),e.begin(),e.end());}
   verify->inputs.at("embeddings")->put(embeddings);elapsed+=verify->run();++physical_calls;
   std::string kn="kv_cache_k_"+std::to_string(layer),vn="kv_cache_v_"+std::to_string(layer);
   auto k=verify->outputs.at(kn)->get(),v=verify->outputs.at(vn)->get();
   auto ks=verify->outputs.at(kn)->type.layout.dimensions;int heads=ks[1],dim=ks[3];
   for(size_t q=0;q<count;++q)for(int part=0;part<2;++part)for(int h=0;h<heads;++h)for(int d=0;d<dim;++d){
    size_t off=part?(size_t(h)*dim+d)*capacity+position+q:(size_t(h)*capacity+position+q)*dim+d;
    const auto& src=part?v:k;float value;std::memcpy(&value,src.data()+off*4,4);require(std::isfinite(value),"Nonfinite verify candidate K/V");packed.insert(packed.end(),src.begin()+off*4,src.begin()+(off+1)*4);
   }
  }
  return {packed,elapsed};
 }
 std::pair<Bytes,double> forward(int id,bool commit){
  return forward_embedding(embedding(id),id,commit,{rope_zero?0:position+rope_delta,rope_zero?0:position+rope_delta,rope_zero?0:position+rope_delta});
 }
 std::pair<Bytes,double> forward_embedding(const Bytes& e,int id,bool commit,const std::vector<int32_t>& rope){
  require(position<capacity&&rope.size()==3,"Invalid decode position");dec->inputs.at("embeddings")->put(e);
  dec->inputs.at("input_pos")->put(bytes<int32_t>({position}));dec->inputs.at("rope_pos")->put(bytes(rope));
  std::vector<float> mask(capacity,-0.7f*std::numeric_limits<float>::max());std::fill(mask.begin(),mask.begin()+position+1,0.f);dec->inputs.at("mask")->put(bytes(mask));
  double seconds=dec->run();Bytes packed;
  for(auto part:{"k","v"}){
   auto& b=*dec->outputs.at("kv_cache_"+std::string(part)+"_"+std::to_string(layer));auto all=b.get();auto shape=b.type.layout.dimensions;
   int heads=shape[1],dim=part[0]=='k'?shape[3]:shape[2];
   for(int h=0;h<heads;h++)for(int d=0;d<dim;d++){
    size_t offset=part[0]=='k'?(size_t(h)*capacity+position)*dim+d:(size_t(h)*dim+d)*capacity+position;
    packed.insert(packed.end(),all.begin()+offset*4,all.begin()+(offset+1)*4);
   }
  }
  if(commit){for(auto& [n,b]:dec->inputs)if(n.rfind("kv_cache_",0)==0)b->put(dec->outputs.at(n)->get());logits=dec->outputs.at("logits")->get();prefix_ids.push_back(id);++position;}
  return {packed,seconds};
 }
#include "qwen_image_prefix.inc"
};
static int serve_qwen(const std::vector<std::string>& a,const J& job){
 if(job.value("batch_signature",std::string())=="prefill_128"&&!job.value("qwen_recovery",false))return serve_qwen_prefill_capture(a,job);
 QwenWorker w(a,job);SocketRpc rpc;rpc.listen_on(job.at("socket_name"));
 publish(a[2]+"/ready.json",{{"ok",true},{"protocol",2},{"transport","socket"},{"batch_capacity",w.slots},{"physical_batch_capacity",w.query_mode=="verify"?w.verify_slots:1},{"qwen_query_mode",w.query_mode},{"query_mode",w.query_mode},{"image_prefix",true},{"selected_rows_restore",true},{"accepted_state_source","decode"},{"vocab_size",w.vocab},{"context_capacity",w.capacity}});rpc.accept_one();int expected=1;
 while(true){
  J c=rpc.receive();require(c.value("auth",std::string())==job.at("socket_auth").get<std::string>(),"RPC auth failed");J reply={{"seq",c.at("seq")},{"ok",false}};Bytes payload;double started=now(),npu=0;bool stop=false;int verify_calls=0,decode_calls=0,prefill_calls=0;
  try{
   require(c.at("seq").get<int>()==expected++,"Sequence mismatch");auto op=c.at("op").get<std::string>();
   if(op=="reset"){w.reset();for(int id:c.at("prefix_ids")){npu+=w.forward(id,true).second;++decode_calls;}}
   else if(op=="reset_image"){npu=w.reset_image(c);decode_calls=w.position;}
   else if(op=="restore_selected_rows"){w.restore_selected_rows(c);}
   else if(op=="diagnostic_rewind"){w.position=0;w.logits.clear();}
   else if(op=="diagnostic_rope_zero"){w.rope_zero=c.at("enabled");}
   else if(op=="query_mode"){
    require(job.value("qwen_recovery",false),"Explicit Qwen recovery mode required");
    auto mode=c.at("mode").get<std::string>();require(mode=="decode"||mode=="exact_prefill"||mode=="verify","Unknown query mode");
    require(mode!="verify"||bool(w.verify),"Verify must be enabled in job.qwen_query_mode at worker creation");
    w.query_mode=mode;w.exact_prefill=mode=="exact_prefill";
   }
   else if(op=="query"){
    require(c.at("position")==w.position,"Position mismatch");auto ids=c.at("ids").get<std::vector<int>>();require(!ids.empty()&&ids.size()<=size_t(w.slots),"Bad query count");
    if(w.query_mode=="verify"){auto result=w.verify_candidates(ids,verify_calls);payload=std::move(result.first);npu=result.second;}
    else for(int id:ids){auto result=w.exact_prefill?w.prefill_candidate(id):w.forward(id,false);payload.insert(payload.end(),result.first.begin(),result.first.end());npu+=result.second;if(w.exact_prefill)prefill_calls+=(w.position+1+127)/128;else ++decode_calls;}
   }else if(op=="causal_prefill"){
                                                                        
                                                                          
    auto ids=c.at("ids").get<std::vector<int>>();
    require(!ids.empty()&&ids.size()<=size_t(w.capacity),"Invalid causal draft sequence");
    if(!w.capture)w.capture=std::make_unique<QwenPrefillCapture>(a,job);
    auto result=w.capture->collect_sequence(ids,c.value("padding_token_id",0));
    payload=std::move(result.first);npu=result.second;
    prefill_calls=(ids.size()+127)/128;
   }else if(op=="advance_tokens"){
    require(c.at("position")==w.position,"Accepted prefix position mismatch");
    auto ids=c.at("ids").get<std::vector<int>>();
    require(w.position+ids.size()<=size_t(w.capacity),"Accepted prefix exceeds capacity");
    for(int id:ids)require(id>=0&&size_t(id)<w.vocab,"Invalid accepted token");
    for(int id:ids){npu+=w.forward(id,true).second;++decode_calls;}
    w.last_exact_row.clear();w.last_exact_id=-1;w.last_exact_position=-1;
    reply["advanced_tokens"]=ids.size();
   }else if(op=="reference_prefill"){
    require(c.at("position")==w.position,"Reference position mismatch");
    auto result=w.reference_prefill(c.at("ids").get<std::vector<int>>());payload=std::move(result.first);npu=result.second;++prefill_calls;
   }else if(op=="commit_cached"){
    require(job.value("qwen_recovery",false)&&w.exact_prefill,"Cached commit requires Qwen exact-prefill mode");
    int id=c.at("id");
    require(c.at("position")==w.position&&w.last_exact_position==w.position&&w.last_exact_id==id&&!w.last_exact_row.empty(),"No matching exact query at current prefix");
    payload=w.last_exact_row;
    auto decode=w.forward(id,true);npu=decode.second;++decode_calls;
    w.last_exact_row.clear();w.last_exact_id=-1;w.last_exact_position=-1;
   }else if(op=="commit"){
    require(c.at("position")==w.position,"Position mismatch");
    if(w.exact_prefill){auto result=w.prefill_candidate(c.at("id"));payload=std::move(result.first);npu=result.second;prefill_calls+=(w.position+1+127)/128;auto decode=w.forward(c.at("id"),true);npu+=decode.second;++decode_calls;}
    else{auto result=w.forward(c.at("id"),true);payload=std::move(result.first);npu=result.second;++decode_calls;}
   }
   else if(op=="embedding")payload=w.embedding(c.at("id"));
   else if(op=="probe_embeddings"){
    auto ids=c.at("ids").get<std::vector<int>>();require(!ids.empty()&&ids.size()<=128,"Embedding probe needs 1..128 IDs");
    size_t width=0;for(int id:ids){auto e=w.embedding(id);require(e.size()%4==0,"Invalid embedding element width");if(!width)width=e.size()/4;require(e.size()==width*4,"Inconsistent embedding width");payload.insert(payload.end(),e.begin(),e.end());}
    reply["embedding_rows"]=ids.size();reply["embedding_width"]=width;reply["embedding_dtype"]="float32";reply["physical_embedder_calls"]=ids.size();
   }
   else if(op=="ranking"){
    require(!w.logits.empty(),"No prefix logits");std::vector<float> score(w.vocab);std::memcpy(score.data(),w.logits.data(),w.logits.size());std::vector<int32_t> ids(w.vocab);std::iota(ids.begin(),ids.end(),0);
    for(float s:score)require(std::isfinite(s),"Nonfinite logits");std::stable_sort(ids.begin(),ids.end(),[&](int x,int y){return score[x]>score[y];});payload=bytes(ids);
   }else if(op=="stop")stop=true;else throw std::runtime_error("Unsupported Qwen operation");
   reply.update({{"ok",true},{"position",w.position},{"payload_bytes",payload.size()},{"decoder_seconds",npu},{"worker_seconds",now()-started},{"query_mode",w.query_mode},{"physical_verify_calls",verify_calls},{"physical_decode_calls",decode_calls},{"physical_prefill_calls",prefill_calls},{"physical_calls",verify_calls+decode_calls+prefill_calls}});
  }catch(const std::exception& e){payload.clear();reply["payload_bytes"]=0;reply["error"]=e.what();}
  rpc.reply(reply,payload);if(stop)return 0;
 }
}
