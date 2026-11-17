                                                                             
                                                                                
                                                                                
#pragma once
struct QwenPrefillCapture {
 LiteRtEnvironment env=nullptr;
 std::unique_ptr<Section> decoder,embedder;
 std::unique_ptr<Signature> prefill,emb;
 int layer=3,slots=128,capacity=2048,position=0;size_t vocab=0;
 std::map<int,Bytes> embeddings;
 QwenPrefillCapture(const std::vector<std::string>& a,const J& job){
  layer=job.at("layer");auto meta=job.at("model_sections");
  void* vendor=dlopen((a[1]+"/libedgetpu_litert.so").c_str(),RTLD_NOW|RTLD_GLOBAL);require(vendor,"Vendor preload failed");
  void* bridge=dlopen((a[1]+"/libannoymous.so").c_str(),RTLD_NOW);require(bridge,"Bridge load failed");
  auto fix=reinterpret_cast<int(*)(void*)>(dlsym(bridge,"ghost_fix_vendor_bulk_io"));require(fix,"Vendor I/O fix absent");fix(dlsym(vendor,"thrGetVendorApiVersion"));
  setenv("GHOST_LAZY_NPU_LOAD","1",1);
  LiteRtEnvOption o{};o.tag=kLiteRtEnvOptionTagDispatchLibraryDir;o.value.type=kLiteRtAnyTypeString;o.value.str_value=a[1].c_str();CHECK(LiteRtCreateEnvironment(1,&o,&env));
  int fd=open(a[0].c_str(),O_RDONLY);require(fd>=0,"Model open failed");
  decoder=std::make_unique<Section>(env,fd,meta.at("tf_lite_prefill_decode"),true,"prefill_128");
  embedder=std::make_unique<Section>(env,fd,meta.at("tf_lite_embedder"),false,"decode_embedder");close(fd);
  prefill=std::make_unique<Signature>(*decoder,"prefill_128");emb=std::make_unique<Signature>(*embedder,"decode_embedder");
  require(prefill->inputs.at("embeddings")->type.layout.dimensions[1]==128,"Expected prefill_128");
  capacity=prefill->inputs.at("kv_cache_k_"+std::to_string(layer))->type.layout.dimensions[2];
  auto decode=meta.at("tf_lite_prefill_decode").at("signatures").at("decode");
  for(auto& t:decode.at("outputs"))if(t.at("name")=="logits"){vocab=1;for(int d:t.at("shape"))vocab*=d;}
  require(vocab>0,"Vocabulary size missing");reset();
 }
 void reset(){
  for(auto& [name,b]:prefill->inputs)if(name.rfind("kv_cache_",0)==0)b->put(Bytes(b->count*b->elem,0));
  position=0;
 }
 const Bytes& embedding(int id){
  require(id>=0&&size_t(id)<vocab,"Invalid token ID");
  auto at=embeddings.find(id);if(at!=embeddings.end())return at->second;
                                                                               
  if(embeddings.size()>=1024)embeddings.clear();
  emb->inputs.at("token_ids")->put(bytes<int32_t>({id}));emb->run();
  return embeddings.emplace(id,emb->outputs.at("embeddings")->get()).first->second;
 }
 void carry_full_block(){
  require(position>0&&position%slots==0,"Cannot carry padding-contaminated recurrent state");
  for(auto& [name,b]:prefill->inputs)if(name.rfind("kv_cache_",0)==0)b->put(prefill->outputs.at(name)->get());
 }
 std::pair<Bytes,double> collect_sequence(const std::vector<int>& ids,int pad_id){
  require(!ids.empty()&&ids.size()<=size_t(capacity),"Invalid causal sequence");reset();
  Bytes packed;double seconds=0;
  for(size_t start=0;start<ids.size();start+=slots){
   if(start)carry_full_block();
   auto end=std::min(ids.size(),start+size_t(slots));
   auto result=collect(std::vector<int>(ids.begin()+start,ids.begin()+end),pad_id,int(start),start==0);
   packed.insert(packed.end(),result.first.begin(),result.first.end());seconds+=result.second;
  }
  return {packed,seconds};
 }
 std::pair<Bytes,double> collect(const std::vector<int>& ids,int pad_id,int start=0,bool clear_state=true){
  require(!ids.empty()&&ids.size()<=size_t(slots)&&start>=0&&start+slots<=capacity,"Invalid prefill block");
                                                                             
                                                                         
  if(clear_state){require(start==0,"Zero-state acquisition must start at zero");reset();}
  else position=start;
  Bytes input;input.reserve(prefill->inputs.at("embeddings")->count*4);
  for(int q=0;q<slots;q++){const auto& row=embedding(q<int(ids.size())?ids[q]:pad_id);input.insert(input.end(),row.begin(),row.end());}
  prefill->inputs.at("embeddings")->put(input);
  std::vector<int32_t> positions(slots),rope(3*slots);std::iota(positions.begin(),positions.end(),start);
  for(int axis=0;axis<3;axis++)std::copy(positions.begin(),positions.end(),rope.begin()+axis*slots);
  prefill->inputs.at("input_pos")->put(bytes(positions));prefill->inputs.at("rope_pos")->put(bytes(rope));
  std::vector<float> mask(size_t(slots)*capacity,-0.7f*std::numeric_limits<float>::max());
  for(int q=0;q<slots;q++)std::fill(mask.begin()+size_t(q)*capacity,mask.begin()+size_t(q)*capacity+start+q+1,0.f);
  prefill->inputs.at("mask")->put(bytes(mask));double seconds=prefill->run();
  auto& kb=*prefill->outputs.at("kv_cache_k_"+std::to_string(layer));
  auto& vb=*prefill->outputs.at("kv_cache_v_"+std::to_string(layer));
  require(kb.elem==4&&vb.elem==4,"Expected native float32 K/V");
  auto k=kb.get(),v=vb.get();Bytes packed;
  for(size_t q=0;q<ids.size();q++)for(int part=0;part<2;part++){
   auto& b=part?vb:kb;auto& all=part?v:k;auto shape=b.type.layout.dimensions;
   int heads=shape[1],dim=part?shape[2]:shape[3];
   for(int h=0;h<heads;h++)for(int d=0;d<dim;d++){
    size_t index=part?(size_t(h)*dim+d)*capacity+start+q:(size_t(h)*capacity+start+q)*dim+d;
    packed.insert(packed.end(),all.begin()+index*4,all.begin()+(index+1)*4);
   }
  }
  position=start+ids.size();return {packed,seconds};
 }
};
static int serve_qwen_prefill_capture(const std::vector<std::string>& a,const J& job){
 QwenPrefillCapture w(a,job);SocketRpc rpc;rpc.listen_on(job.at("socket_name"));
 publish(a[2]+"/ready.json",{{"ok",true},{"protocol",2},{"transport","socket"},{"batch_capacity",128},{"vocab_size",w.vocab},{"context_capacity",w.capacity},{"acquisition_only",true},{"signature","prefill_128"},{"independent_zero_kvcr",true}});
 rpc.accept_one();int expected=1;
 while(true){
  J c=rpc.receive();require(c.value("auth",std::string())==job.at("socket_auth").get<std::string>(),"RPC auth failed");
  J reply={{"seq",c.at("seq")},{"ok",false}};Bytes payload;double started=now(),npu=0;bool stop=false;
  try{
   require(c.at("seq").get<int>()==expected++,"Sequence mismatch");auto op=c.at("op").get<std::string>();
   if(op=="reset"){require(c.at("prefix_ids").empty(),"Acquisition reset requires an empty prefix");w.reset();}
   else if(op=="collect_tokens"){
    require(c.value("position",w.position)==w.position,"Collection position mismatch");
    int start=w.position;if(start)w.carry_full_block();
    auto result=w.collect(c.at("ids").get<std::vector<int>>(),c.value("padding_token_id",0),start,start==0);payload=std::move(result.first);npu=result.second;
   }else if(op=="stop")stop=true;else throw std::runtime_error("prefill_128 acquisition worker does not support candidate recovery");
   reply.update({{"ok",true},{"position",w.position},{"payload_bytes",payload.size()},{"decoder_seconds",npu},{"worker_seconds",now()-started},{"signature","prefill_128"}});
  }catch(const std::exception& e){payload.clear();reply["payload_bytes"]=0;reply["error"]=e.what();}
  rpc.reply(reply,payload);if(stop)return 0;
 }
}
