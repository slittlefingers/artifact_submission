#include "litert_support.h"
#include "socket_rpc.h"
#include "float_ranking.h"
#include <list>
#include <thread>
#include <sys/stat.h>
#include "image_collection.h"
#include "audio_framework_bridge.h"

static Bytes collection_byte_hex(const std::string& text){
 require(text.size()%2==0&&text.size()<=(2u<<20),"Invalid image upload length");
 auto digit=[](char c)->int{if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;throw std::runtime_error("Invalid image hex");};
 Bytes out(text.size()/2);for(size_t i=0;i<out.size();i++)out[i]=uint8_t(digit(text[2*i])*16+digit(text[2*i+1]));return out;
}

static Bytes finite_float_hex(const std::string& text,size_t floats){
 require(text.size()==floats*sizeof(float)*2,"Invalid float payload length");
 auto digit=[](char c)->int{if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;throw std::runtime_error("Invalid hex digit");};
 Bytes out(floats*sizeof(float));for(size_t i=0;i<out.size();i++)out[i]=uint8_t(digit(text[2*i])*16+digit(text[2*i+1]));
 for(size_t i=0;i<floats;i++){float v;std::memcpy(&v,out.data()+i*sizeof(float),sizeof(float));require(std::isfinite(v),"Nonfinite float input");}return out;
}

                                                                            
                                                                       
struct Worker {
 LiteRtEnvironment env=nullptr;
 std::unique_ptr<Section> decoder,embedding,perlayer,aux;
 std::unique_ptr<Signature> dec,batch,emb,ple,rope,mask;
 std::string model_path;J vision_meta,encoder_meta;
 std::unique_ptr<Section> image_encoder;
 std::unique_ptr<Signature> image_signature;
 std::unique_ptr<Signature> collection_rope,collection_mask;
 std::unique_ptr<Section> vision;
 std::unique_ptr<Signature> vision_adapter;
 std::unique_ptr<FrameworkAudio> framework_audio;
 int position=0,layer=4,slots=128,capacity=4096,mask_token=1437;size_t vocab=0;
 Data logits;
 std::pair<Data,Data> acquisition_prefix_cache;int acquisition_prefix_position=-1;
 std::pair<Data,Data> media_prefix_cache;Data media_prefix_logits;int media_prefix_position=-1;
 std::pair<Data,Data> cached_context;int context_position=-1,batch_context_position=-1;
 std::map<int,Data> embedding_cache;
 std::list<int> embedding_lru;
 std::map<int,std::list<int>::iterator> embedding_lru_index;
 Data batch_embedding_scratch;
 bool retain_candidates=false;
 bool selected_retention=false;
 std::vector<int> last_query_ids;
 int last_query_position=-1;
 bool experimental_query_outputs_valid=false;
 std::map<int,Data> candidate_slices;
 Worker(const std::string& model,const std::string& libs,const J& meta,int l,const std::string& sig,int mask_id):layer(l),mask_token(mask_id){
  model_path=model;if(meta.contains("tf_lite_vision_adapter"))vision_meta=meta.at("tf_lite_vision_adapter");
  if(meta.contains("tf_lite_vision_encoder"))encoder_meta=meta.at("tf_lite_vision_encoder");
  require(l>=0,"Invalid KV producer layer");
  LiteRtEnvOption o{};o.tag=kLiteRtEnvOptionTagDispatchLibraryDir;o.value.type=kLiteRtAnyTypeString;o.value.str_value=libs.c_str();
  CHECK(LiteRtCreateEnvironment(1,&o,&env));int fd=open(model.c_str(),O_RDONLY);require(fd>=0,"Model open failed");
  decoder=std::make_unique<Section>(env,fd,meta.at("tf_lite_prefill_decode"),true);
  embedding=std::make_unique<Section>(env,fd,meta.at("tf_lite_embedder"),false);
  if(meta.contains("tf_lite_per_layer_embedder"))perlayer=std::make_unique<Section>(env,fd,meta.at("tf_lite_per_layer_embedder"),false);
  aux=std::make_unique<Section>(env,fd,meta.at("tf_lite_aux"),false);close(fd);
  dec=std::make_unique<Signature>(*decoder,"decode");batch=std::make_unique<Signature>(*decoder,sig);
  vocab=dec->outputs.at("logits")->count;
  slots=batch->inputs.at("embeddings")->type.layout.dimensions[1];
  capacity=dec->inputs.at("kv_cache_k_"+std::to_string(layer))->type.layout.dimensions[2];
  require(slots>0&&capacity>0&&mask_token>=0&&size_t(mask_token)<vocab,"Invalid compiled dimensions or mask token");
  for(auto part:{"k","v"})require(batch->outputs.count("kv_slice_"+std::string(part)+"_"+std::to_string(layer)),"Selected KV layer missing");
  emb=std::make_unique<Signature>(*embedding,"decode_embedder");if(perlayer)ple=std::make_unique<Signature>(*perlayer,"decode_per_layer_embedder");
  rope=std::make_unique<Signature>(*aux,"decode_rope");mask=std::make_unique<Signature>(*aux,"decode_mask");
  reset();
 }
 void reset(){
  for(auto* s:{dec.get(),batch.get()})for(auto& [n,b]:s->inputs)if(n.rfind("kv_cache_",0)==0)b->put(Bytes(b->count*b->elem,0));
  context_position=-1;batch_context_position=-1;cached_context={};
  position=0;logits.clear();embedding_cache.clear();embedding_lru.clear();embedding_lru_index.clear();batch_embedding_scratch.clear();candidate_slices.clear();retain_candidates=false;experimental_query_outputs_valid=false;
 }
                                                                              
 void save_acquisition_prefix(){
  require(position>0,"Empty acquisition prefix");
  acquisition_prefix_cache={};
  for(auto& [n,b]:dec->inputs)if(n.rfind("kv_cache_",0)==0)acquisition_prefix_cache.first[n]=b->get();
  for(auto& [n,b]:batch->inputs)if(n.rfind("kv_cache_",0)==0)acquisition_prefix_cache.second[n]=b->get();
  acquisition_prefix_position=position;
 }
 void restore_acquisition_prefix(){
  require(acquisition_prefix_position>0&&!acquisition_prefix_cache.first.empty(),"No acquisition prefix saved");
  dec->put(acquisition_prefix_cache.first);batch->put(acquisition_prefix_cache.second);
  position=acquisition_prefix_position;context_position=-1;batch_context_position=-1;cached_context={};
  candidate_slices.clear();retain_candidates=false;logits.clear();experimental_query_outputs_valid=false;
 }
 const Data& embed(int id){
  require(id>=0&&size_t(id)<vocab,"Token ID outside original vocabulary");
  auto found=embedding_cache.find(id);
  if(found!=embedding_cache.end()){
   profile_counts["embedding_hit"]++;
   if(native_optimized_v2)embedding_lru.splice(embedding_lru.begin(),embedding_lru,embedding_lru_index.at(id));
   return found->second;
  }
  profile_counts["embedding_miss"]++;
                                                              
  if(embedding_cache.size()>=1024){
   if(native_optimized_v2){int oldest=embedding_lru.back();embedding_lru.pop_back();embedding_lru_index.erase(oldest);embedding_cache.erase(oldest);profile_counts["embedding_evictions"]++;}
   else embedding_cache.clear();
  }
  auto token=bytes<int32_t>({id});emb->inputs.at("token_ids")->put(token);emb->run();Data d={{"embeddings",emb->outputs.at("embeddings")->get()}};
  if(ple){ple->inputs.at("token_ids")->put(token);ple->run();d["per_layer_embeddings"]=ple->outputs.at("embeddings")->get();}
  auto& result=embedding_cache.emplace(id,std::move(d)).first->second;
  if(native_optimized_v2){embedding_lru.push_front(id);embedding_lru_index[id]=embedding_lru.begin();}
  return result;
 }
 const std::pair<Data,Data>& context(){
  require(position<capacity,"Compiled context capacity exhausted");
  if(native_optimized&&context_position==position){profile_counts["context_hit"]++;return cached_context;}
  profile_counts["context_miss"]++;
  rope->inputs.at("input_pos")->put(bytes<int32_t>({position}));rope->run();
  mask->inputs.at("input_tokens")->put(bytes<int32_t>({mask_token}));mask->inputs.at("time_step")->put(bytes<int32_t>({position}));mask->run();
  cached_context={rope->get(),mask->get()};context_position=position;return cached_context;
 }
 void prepare_batch_context(){
  if(native_optimized&&batch_context_position==position){profile_counts["batch_context_hit"]++;return;}
  profile_counts["batch_context_miss"]++;
  const auto& [r,m]=context();Data input;
  for(auto& [n,b]:r)input[n]=repeat(b,slots);
  for(auto& [n,b]:m){require(b.size()==size_t(capacity+1)*sizeof(float),"Unexpected mask");std::vector<float> base(capacity+1);std::memcpy(base.data(),b.data(),b.size());
   require(base[capacity-1]!=base[capacity],"Unable to identify hidden mask value");int columns=capacity+slots;std::vector<float> branched(slots*columns,base[capacity-1]);
   for(int q=0;q<slots;q++){std::copy(base.begin(),base.begin()+capacity,branched.begin()+q*columns);branched[q*columns+capacity+q]=base[capacity];}
   input[n]=bytes(branched);
  }
  batch->put(input);batch_context_position=position;
 }
 void prepare_decode(int id){
  const auto& [r,m]=context();dec->put(r);dec->put(m);
  if(native_optimized)dec->put(embed(id));else{Data copy=embed(id);dec->put(copy);}
 }
 Data selected(Signature& s){Data d;for(auto part:{"k","v"}){auto n="kv_slice_"+std::string(part)+"_"+std::to_string(layer);d[n]=s.outputs.at(n)->get();}return d;}
 Bytes packed(const Data& d){Bytes v;for(auto part:{"k","v"}){auto& b=d.at("kv_slice_"+std::string(part)+"_"+std::to_string(layer));v.insert(v.end(),b.begin(),b.end());}return v;}
                                                                                   
 void append_at(Signature& s,const Data& slices,int at){
  for(auto& [name,src]:slices){std::string n=name;n.replace(0,9,"kv_cache_");if(!s.inputs.count(n))continue;
   auto& b=*s.inputs.at(n);int axis=n.rfind("kv_cache_k_",0)==0?2:3;
   require(at>=0&&at<b.type.layout.dimensions[axis],"Cache row outside compiled capacity");
   size_t inner=1,outer=1;for(unsigned i=axis+1;i<b.type.layout.rank;i++)inner*=b.type.layout.dimensions[i];for(int i=0;i<axis;i++)outer*=b.type.layout.dimensions[i];
   require(src.size()==outer*inner*b.elem,"Unexpected cache slice layout");void* p;CHECK(LiteRtLockTensorBuffer(b.h,&p,kLiteRtTensorBufferLockModeWrite));
   for(size_t o=0;o<outer;o++){
    size_t linear=(o*b.type.layout.dimensions[axis]+at)*inner;
    b.copier->copy(p,const_cast<uint8_t*>(src.data()+o*inner*b.elem),linear,inner,true,native_optimized);
   }
   CHECK(LiteRtUnlockTensorBuffer(b.h));
  }
 }
 void append(Signature& s,const Data& slices){append_at(s,slices,position);}
                                                                         
                                                                         
 std::vector<std::string> experimental_cache_names(const std::vector<int>& layers){
  require(!layers.empty()&&layers.size()<=15,"Invalid experimental layer count");
  std::vector<std::string> names;std::vector<int> seen;
  for(int l:layers){
   require(l>=0&&std::find(seen.begin(),seen.end(),l)==seen.end(),"Invalid or duplicate experimental layer");seen.push_back(l);
   for(auto part:{"k","v"}){
    std::string name="kv_cache_"+std::string(part)+"_"+std::to_string(l);
    require(dec->inputs.count(name),"Missing experimental decode cache layer");
    auto& d=*dec->inputs.at(name);int axis=std::string(part)=="k"?2:3;
    require(d.type.layout.rank==4,"Unsupported experimental cache rank");
    require(d.type.element_type==kLiteRtElementTypeInt8||d.type.element_type==kLiteRtElementTypeFloat32,"Unsupported experimental cache dtype");
                                                                      
    if(batch->inputs.count(name)){
     auto& b=*batch->inputs.at(name);require(b.type.layout.rank==4&&d.type.element_type==b.type.element_type,"Experimental cache rank/dtype mismatch");
     for(unsigned i=0;i<4;i++)require(d.type.layout.dimensions[i]==b.type.layout.dimensions[i],"Experimental cache shape mismatch");
    }
    require(d.type.layout.dimensions[axis]==capacity,"Unsupported experimental cache sequence capacity");
    names.push_back(name);
   }
   auto& kn=names[names.size()-2];auto& vn=names.back();
   require(dec->inputs.at(kn)->type.element_type==dec->inputs.at(vn)->type.element_type,"Experimental K/V dtype mismatch");
   require(batch->inputs.count(kn)==batch->inputs.count(vn),"Experimental prefill K/V presence mismatch");
  }
  return names;
 }
 size_t experimental_row_bytes(const std::vector<std::string>& names){
  size_t total=0;for(auto& name:names){auto& b=*dec->inputs.at(name);total+=b.count/size_t(capacity)*b.elem;}return total;
 }
 J experimental_layer_layout(const std::vector<int>& layers){
  auto names=experimental_cache_names(layers);J result=J::array();
  for(size_t i=0;i<layers.size();i++){
   auto& k=*dec->inputs.at(names[2*i]);auto& v=*dec->inputs.at(names[2*i+1]);
   result.push_back({{"layer",layers[i]},{"k_bytes",k.count/size_t(capacity)*k.elem},{"v_bytes",v.count/size_t(capacity)*v.elem},{"dtype",k.elem==1?"int8":"float32"},{"prefill_cache_present",batch->inputs.count(names[2*i])!=0}});
  }return result;
 }
 void experimental_past_range(int start,int rows){
  require(start>=0&&rows>0&&rows<=128&&start<=position&&rows<=position-start,"Experimental range must contain 1..128 past rows only");
 }
 Bytes experimental_read_past_kv(const std::vector<int>& layers,int start,int rows,bool prefill){
  experimental_past_range(start,rows);auto names=experimental_cache_names(layers);Bytes result;
  result.reserve(size_t(rows)*experimental_row_bytes(names));auto& signature=prefill?*batch:*dec;
  for(auto& name:names)require(signature.inputs.count(name),"Requested cache source omits this layer");
  for(int q=0;q<rows;q++)for(auto& name:names){
   auto value=signature.inputs.at(name)->get_row(name.rfind("kv_cache_k_",0)==0?2:3,start+q,capacity);
   result.insert(result.end(),value.begin(),value.end());
  }return result;
 }
 void experimental_patch_past_kv(const std::vector<int>& layers,int start,int rows,const Bytes& payload){
  experimental_past_range(start,rows);auto names=experimental_cache_names(layers);
  require(payload.size()==size_t(rows)*experimental_row_bytes(names),"Experimental KV payload length mismatch");
                                                                             
  std::vector<Data> patches(rows);size_t offset=0;
  for(int q=0;q<rows;q++)for(auto& name:names){
   auto& b=*dec->inputs.at(name);size_t length=b.count/size_t(capacity)*b.elem;
   if(b.elem==4)for(size_t i=0;i<length;i+=4){float value;std::memcpy(&value,payload.data()+offset+i,4);require(std::isfinite(value),"Nonfinite experimental KV value");}
   std::string slice=name;slice.replace(0,9,"kv_slice_");
   patches[q][slice]=Bytes(payload.begin()+offset,payload.begin()+offset+length);offset+=length;
  }
                                                                              
  candidate_slices.clear();retain_candidates=false;selected_retention=false;
  last_query_ids.clear();last_query_position=-1;experimental_query_outputs_valid=false;
  acquisition_prefix_cache={};acquisition_prefix_position=-1;logits.clear();
  cached_context={};context_position=-1;batch_context_position=-1;
  for(int q=0;q<rows;q++){append_at(*dec,patches[q],start+q);append_at(*batch,patches[q],start+q);}
 }
 Bytes experimental_read_query_kv(const std::vector<int>& layers,const std::vector<int>& ids){
  require(experimental_query_outputs_valid&&last_query_position==position&&!ids.empty()&&ids==last_query_ids,"Experimental query read must match the latest query IDs and position");
  auto names=experimental_cache_names(layers);std::vector<std::string> slices;Bytes result;
  for(auto name:names){
   name.replace(0,9,"kv_slice_");require(batch->outputs.count(name),"Missing experimental query layer");
   auto& b=*batch->outputs.at(name);int axis=name.rfind("kv_slice_k_",0)==0?2:3;
   require(b.type.layout.rank==4&&b.type.layout.dimensions[axis]==slots,"Unsupported experimental query layout");
   std::string cache=name;cache.replace(0,9,"kv_cache_");auto& d=*dec->inputs.at(cache);
   require(b.type.element_type==d.type.element_type&&b.count/size_t(slots)==d.count/size_t(capacity),"Experimental query/cache layout mismatch");
   for(unsigned i=0;i<4;i++)if(int(i)!=axis)require(b.type.layout.dimensions[i]==d.type.layout.dimensions[i],"Experimental query/cache row shape mismatch");
   slices.push_back(name);
  }
  result.reserve(ids.size()*experimental_row_bytes(names));
  for(size_t q=0;q<ids.size();q++)for(auto& name:slices){
   auto value=batch->outputs.at(name)->get_row(name.rfind("kv_slice_k_",0)==0?2:3,int(q),slots);
   result.insert(result.end(),value.begin(),value.end());
  }return result;
 }
 std::pair<Bytes,double> commit(int id){
  candidate_slices.clear();retain_candidates=false;
  double tick=now();prepare_decode(id);measured("phase_commit_inputs",tick);
  double t=dec->run();
  tick=now();auto kv=dec->get(true);auto result=packed(kv);logits={{"logits",dec->outputs.at("logits")->get()}};measured("phase_commit_outputs",tick);
  tick=now();append(*dec,kv);append(*batch,kv);measured("phase_commit_append",tick);
  ++position;return {result,t};
 }
                                                                              
                                                                        
 std::pair<Bytes,double> commit_embedding(const std::vector<float>& values,int ple_id){
  auto& input=*dec->inputs.at("embeddings");
  require(input.elem==sizeof(float)&&values.size()==input.count,"Invalid media embedding shape");
  for(float x:values)require(std::isfinite(x),"Nonfinite media embedding");
  candidate_slices.clear();retain_candidates=false;
  const auto& [r,m]=context();dec->put(r);dec->put(m);
  dec->put(embed(ple_id));input.put(bytes(values));
  double t=dec->run();auto kv=dec->get(true);auto result=packed(kv);
  logits={{"logits",dec->outputs.at("logits")->get()}};
  append(*dec,kv);append(*batch,kv);++position;return {result,t};
 }
                                                                               
                                                                               
 std::pair<Bytes,double> collect_embeddings(const Bytes& values,int rows){
  require(rows>0&&rows<=slots&&position+rows<=capacity,"Invalid collection block");
  require(values.size()==size_t(rows)*1536*4,"Invalid collection embedding width");
  if(!collection_rope){
   collection_rope=std::make_unique<Signature>(*aux,"prefill_rope_"+std::to_string(slots));
   collection_mask=std::make_unique<Signature>(*aux,"prefill_mask_"+std::to_string(slots));
  }
  candidate_slices.clear();retain_candidates=false;experimental_query_outputs_valid=false;
  std::vector<int32_t> positions(slots),tokens(slots,mask_token);
  std::iota(positions.begin(),positions.end(),position);
  collection_rope->inputs.at("input_pos")->put(bytes(positions));collection_rope->run();
  collection_mask->inputs.at("input_tokens")->put(bytes(tokens));
  collection_mask->inputs.at("time_step")->put(bytes<int32_t>({position}));collection_mask->run();
  batch->put(collection_rope->get());batch->put(collection_mask->get());
  batch->put({{"per_layer_embeddings",repeat(embed(0).at("per_layer_embeddings"),slots)}});
  Bytes padded(size_t(slots)*1536*4,0);std::copy(values.begin(),values.end(),padded.begin());
  batch->inputs.at("embeddings")->put(padded);
  double elapsed=batch->run();auto all=batch->get(true);Bytes output;
  for(int q=0;q<rows;q++){
   auto one=row(*batch,all,q,slots);auto code=packed(one);
   output.insert(output.end(),code.begin(),code.end());
   append(*dec,one);append(*batch,one);++position;
  }
  batch_context_position=-1;context_position=-1;logits.clear();
  return {output,elapsed};
 }
 Bytes project_vision(const Bytes& values,int rows){
  require(rows>0&&rows<=slots&&rows<=128&&values.size()==size_t(rows)*768*sizeof(float),"Invalid vision candidate shape");
  if(!vision_adapter){
   require(!vision_meta.is_null(),"Vision adapter absent");int fd=open(model_path.c_str(),O_RDONLY);require(fd>=0,"Vision model open failed");
   try{vision=std::make_unique<Section>(env,fd,vision_meta,false);}catch(...){close(fd);throw;}close(fd);
   vision_adapter=std::make_unique<Signature>(*vision,"vision_adapter_280");
  }
  auto& in=*vision_adapter->inputs.at("soft_tokens");auto& out=*vision_adapter->outputs.at("mm_embedding");
  require(in.count==280*768&&out.count==280*1536&&in.elem==4&&out.elem==4,"Unsupported vision adapter layout");
  Bytes padded(in.count*in.elem,0);std::copy(values.begin(),values.end(),padded.begin());in.put(padded);vision_adapter->run();
  auto result=out.get();result.resize(size_t(rows)*1536*sizeof(float));return result;
 }
 std::pair<Bytes,double> query_embeddings(const Bytes& values,int rows){
  require(rows>0&&rows<=slots&&values.size()==size_t(rows)*1536*sizeof(float),"Invalid continuous candidate shape");
  candidate_slices.clear();retain_candidates=false;experimental_query_outputs_valid=false;prepare_batch_context();Data input;
  const auto& e=embed(0);for(auto& [n,b]:e)input[n]=repeat(b,slots);
  auto& dest=input.at("embeddings");require(dest.size()==size_t(slots)*1536*sizeof(float),"Unsupported embedding layout");
  for(int q=0;q<slots;q++)std::memcpy(dest.data()+q*1536*sizeof(float),values.data()+(q<rows?q:0)*1536*sizeof(float),1536*sizeof(float));
  batch->put(input);double t=batch->run();auto data=selected(*batch);Bytes output;
  for(int q=0;q<rows;q++){auto v=packed(row(*batch,data,q,slots));output.insert(output.end(),v.begin(),v.end());}return {output,t};
 }
 void retain_selected(int id){
  require(retain_candidates&&selected_retention&&last_query_position==position,"No live query output for selection");
  auto it=std::find(last_query_ids.begin(),last_query_ids.end(),id);
  require(it!=last_query_ids.end(),"Selected token is not in the latest query");
  int q=int(it-last_query_ids.begin());Data chosen;double tick=now();
  for(auto& [name,b]:batch->outputs)if(name.rfind("kv_slice_",0)==0)
   chosen[name]=b->get_row(name.rfind("kv_slice_k_",0)==0?2:3,q,slots);
  candidate_slices.clear();candidate_slices.emplace(id,std::move(chosen));
  measured("phase_retain_selected",tick);profile_counts["retained_candidates"]++;
 }
 std::pair<Bytes,double> accept_query(int id){
  require(retain_candidates&&candidate_slices.count(id),"Candidate cache not retained");
                                                                                  
  double tick=now();prepare_decode(id);measured("phase_commit_inputs",tick);
  double t=dec->run();auto& kv=candidate_slices.at(id);
  tick=now();auto result=packed(kv);logits={{"logits",dec->outputs.at("logits")->get()}};measured("phase_commit_outputs",tick);
  tick=now();append(*dec,kv);append(*batch,kv);measured("phase_commit_append",tick);
  ++position;candidate_slices.clear();retain_candidates=false;return {result,t};
 }
 std::pair<Bytes,double> query(const std::vector<int>& ids){
  require(!ids.empty()&&ids.size()<=size_t(slots),"Query exceeds compiled candidate slots");
  experimental_query_outputs_valid=false;
  double tick=now();prepare_batch_context();measured("phase_query_context",tick);
  tick=now();Data temporary;Data& input=native_optimized_v2?batch_embedding_scratch:temporary;
  for(int q=0;q<slots;q++){
   int id=q<int(ids.size())?ids[q]:ids[0];
   if(native_optimized_v2){
    const auto& e=embed(id);
    for(auto& [n,b]:e){auto& dest=input[n];if(dest.size()!=b.size()*slots)dest.resize(b.size()*slots);std::memcpy(dest.data()+q*b.size(),b.data(),b.size());}
   }
   else if(native_optimized){const auto& e=embed(id);for(auto& [n,b]:e){auto& dest=input[n];if(q==0)dest.reserve(b.size()*slots);dest.insert(dest.end(),b.begin(),b.end());}}
   else{Data e=embed(id);for(auto& [n,b]:e)input[n].insert(input[n].end(),b.begin(),b.end());}
  }
  measured("phase_query_embedding",tick);
  tick=now();batch->put(input);measured("phase_query_write",tick);
  double t=batch->run();
  last_query_ids=ids;last_query_position=position;
  bool eager=retain_candidates&&!selected_retention;
  tick=now();auto data=eager?batch->get(true):selected(*batch);measured("phase_query_read",tick);
  tick=now();Bytes output;
  if(eager){
   require(candidate_slices.size()+ids.size()<=4096,"Retained logical batch exceeds 4096");
   for(size_t q=0;q<ids.size();q++)candidate_slices[ids[q]]=row(*batch,data,q,slots);
  }
  for(size_t q=0;q<ids.size();q++){auto v=(native_optimized&&eager)?packed(candidate_slices.at(ids[q])):packed(row(*batch,data,q,slots));output.insert(output.end(),v.begin(),v.end());}
   measured("phase_query_pack",tick);experimental_query_outputs_valid=true;return {output,t};
 }

};

static void publish(const std::string& path,const J& j){savej(path+".tmp",j);require(rename((path+".tmp").c_str(),path.c_str())==0,"Publish failed");}
#include "qwen_worker.h"
static int serve(const std::vector<std::string>& a){
 try{
  require(a.size()==3,"Expected model, library directory, job directory");auto root=a[2];auto job=J::parse(read(root+"/job.json"));
  if(!job.at("model_sections").contains("tf_lite_aux"))return serve_qwen(a,job);
  auto mode=job.value("native_mode",std::string("optimized_v2"));require(mode=="reference"||mode=="optimized"||mode=="optimized_v2","Invalid native mode");native_optimized=mode!="reference";native_optimized_v2=mode=="optimized_v2";
  Worker w(a[0],a[1],job.at("model_sections"),job.at("layer").get<int>(),job.value("batch_signature","prefill_128"),job.value("mask_token_id",1437));
  const bool socket_mode=job.value("transport","file")=="socket";SocketRpc rpc;
  if(socket_mode)rpc.listen_on(job.at("socket_name"));
   publish(root+"/ready.json",{{"ok",true},{"protocol",2},{"selected_retention",true},{"experimental_past_kv",1},{"experimental_query_kv",1},{"transport",socket_mode?"socket":"file"},{"batch_capacity",w.slots},{"vocab_size",w.vocab},{"context_capacity",w.capacity}});
  if(socket_mode)rpc.accept_one();
  int expected=1;double last=now();
  while(now()-last<900){
   J c;
   if(socket_mode){c=rpc.receive();require(c.value("auth",std::string())==job.at("socket_auth").get<std::string>(),"RPC authentication failed");}
   else{std::string path=root+"/command.json";if(access(path.c_str(),F_OK)!=0){std::this_thread::sleep_for(std::chrono::milliseconds(10));continue;}
    c=J::parse(read(path));unlink(path.c_str());}
   profile_seconds.clear();profile_counts.clear();
   J reply={{"seq",c.at("seq")},{"ok",false}};double started=now();bool stop=false;Bytes payload;
   try{
    require(c.at("seq").get<int>()==expected,"Command sequence mismatch");++expected;auto op=c.at("op").get<std::string>();double npu=0;
    if(op=="reset"){
     w.reset();for(int id:c.at("prefix_ids")){npu+=w.commit(id).second;}
    }else if(op=="save_acquisition_prefix"){
     require(c.value("training_acquisition",false),"Explicit acquisition opt-in required");
     w.save_acquisition_prefix();reply["saved_prefix_position"]=w.acquisition_prefix_position;
    }else if(op=="restore_acquisition_prefix"){
     require(c.value("training_acquisition",false),"Explicit acquisition opt-in required");
     w.restore_acquisition_prefix();
    }else if(op=="save_media_prefix"){
     require(c.at("position").get<int>()==w.position&&w.media_prefix_position<0,"Media snapshot position/nesting mismatch");
     w.media_prefix_cache={};w.media_prefix_logits=w.logits;
     for(auto& [n,b]:w.dec->inputs)if(n.rfind("kv_cache_",0)==0)w.media_prefix_cache.first[n]=b->get();
     for(auto& [n,b]:w.batch->inputs)if(n.rfind("kv_cache_",0)==0)w.media_prefix_cache.second[n]=b->get();
     w.media_prefix_position=w.position;
    }else if(op=="restore_media_prefix"){
     require(w.media_prefix_position>=0,"No media snapshot saved");
     w.dec->put(w.media_prefix_cache.first);w.batch->put(w.media_prefix_cache.second);
     w.position=w.media_prefix_position;w.logits=w.media_prefix_logits;
     w.context_position=-1;w.batch_context_position=-1;w.cached_context={};
     w.candidate_slices.clear();w.retain_candidates=false;w.experimental_query_outputs_valid=false;
    }else if(op=="release_media_prefix"){
     w.media_prefix_cache={};w.media_prefix_logits.clear();w.media_prefix_position=-1;
    }else if(op=="experimental_read_past_kv"||op=="experimental_patch_past_kv"){
     require(c.at("position").get<int>()==w.position,"Experimental cache position mismatch");
     auto layers=c.at("layers").get<std::vector<int>>();int start=c.at("start"),rows=c.at("rows");
     auto layout=w.experimental_layer_layout(layers);
     if(op=="experimental_patch_past_kv"){
      require(c.value("allow_observed_past_kv",false),"Explicit observed past-KV patch opt-in required");
      auto values=collection_byte_hex(c.at("codes_hex").get<std::string>());
      w.experimental_patch_past_kv(layers,start,rows,values);
      reply["patched_sources"]={"decode","present_prefill_inputs"};
     }else{
      auto source=c.value("source",std::string("decode"));require(source=="decode"||source=="prefill","Unknown experimental cache source");
      payload=w.experimental_read_past_kv(layers,start,rows,source=="prefill");reply["source"]=source;
     }
     reply["start"]=start;reply["rows"]=rows;reply["layers"]=layers;reply["layer_layout"]=layout;
     reply["wire_order"]="row, requested layer, packed K then V";
    }else if(op=="experimental_read_query_kv"){
     require(c.at("position").get<int>()==w.position,"Experimental query position mismatch");
     auto layers=c.at("layers").get<std::vector<int>>();auto ids=c.at("ids").get<std::vector<int>>();
     payload=w.experimental_read_query_kv(layers,ids);reply["ids"]=ids;reply["layers"]=layers;
     reply["layer_layout"]=w.experimental_layer_layout(layers);reply["wire_order"]="candidate, requested layer, packed K then V";
    }else if(op=="query"){
     require(c.at("position").get<int>()==w.position,"Prefix position mismatch");
     if(c.value("begin_candidates",false)){
      w.candidate_slices.clear();w.retain_candidates=true;w.selected_retention=c.value("selected_retention",false);w.last_query_ids.clear();w.last_query_position=-1;
     }
     if(c.contains("retain_id"))w.retain_selected(c.at("retain_id"));
     require(c.at("position").get<int>()==w.position,"Prefix position mismatch");auto v=w.query(c.at("ids").get<std::vector<int>>());payload=std::move(v.first);npu=v.second;
    }else if(op=="query_v_layers"||op=="query_kv_layers"){
     require(c.at("position").get<int>()==w.position,"Prefix position mismatch");
     auto ids=c.at("ids").get<std::vector<int>>();
     auto layers=c.at("layers").get<std::vector<int>>();
     require(!layers.empty()&&layers.size()<=15,"Invalid selected layer count");
     w.retain_candidates=false;
     auto result=w.query(ids);npu=result.second;
     Data data;
                                                                        
                                                                          
     const bool include_k=op=="query_kv_layers";
     for(int layer:layers){
      if(include_k){auto name="kv_slice_k_"+std::to_string(layer);require(w.batch->outputs.count(name),"Missing K layer");data[name]=w.batch->outputs.at(name)->get();}
      auto name="kv_slice_v_"+std::to_string(layer);require(w.batch->outputs.count(name),"Missing V layer");data[name]=w.batch->outputs.at(name)->get();
     }
     for(size_t q=0;q<ids.size();q++){
      auto values=row(*w.batch,data,q,w.slots);
      for(int layer:layers){
       if(include_k){auto& bytes=values.at("kv_slice_k_"+std::to_string(layer));payload.insert(payload.end(),bytes.begin(),bytes.end());}
       auto& bytes=values.at("kv_slice_v_"+std::to_string(layer));payload.insert(payload.end(),bytes.begin(),bytes.end());
      }
     }
    }else if(op=="diagnostic_prefill_tokens"||op=="collect_tokens"){
     w.experimental_query_outputs_valid=false;
     auto ids=c.at("ids").get<std::vector<int>>();int count=ids.size();
     require(count>0&&count<=w.slots&&w.position+count<=w.capacity,"Invalid diagnostic prefill");
     Signature r(*w.aux,"prefill_rope_"+std::to_string(w.slots));
     Signature m(*w.aux,"prefill_mask_"+std::to_string(w.slots));
     std::vector<int32_t> positions(w.slots),tokens(w.slots,-1);std::iota(positions.begin(),positions.end(),w.position);
     std::copy(ids.begin(),ids.end(),tokens.begin());
     r.inputs.at("input_pos")->put(bytes(positions));r.run();
     m.inputs.at("input_tokens")->put(bytes(tokens));m.inputs.at("time_step")->put(bytes<int32_t>({w.position}));m.run();
     w.batch->put(r.get());w.batch->put(m.get());Data inputs;
     for(int q=0;q<w.slots;q++){auto data=w.embed(q<count?ids[q]:0);for(auto& [name,value]:data)inputs[name].insert(inputs[name].end(),value.begin(),value.end());}
     w.batch->put(inputs);npu=w.batch->run();auto all=w.batch->get(true);
     auto layers=c.at("layers").get<std::vector<int>>();
     for(int q=0;q<count;q++){auto one=row(*w.batch,all,q,w.slots);for(int layer:layers){
      if(op=="collect_tokens"){auto& key=one.at("kv_slice_k_"+std::to_string(layer));payload.insert(payload.end(),key.begin(),key.end());}
      auto& value=one.at("kv_slice_v_"+std::to_string(layer));payload.insert(payload.end(),value.begin(),value.end());
     }w.append(*w.dec,one);w.append(*w.batch,one);++w.position;}
     w.context_position=-1;w.batch_context_position=-1;w.cached_context={};
     if(op=="collect_tokens")w.logits.clear();
    }else if(op=="prefix_logits"||op=="prefix_embedding_logits"){
                                                                             
                                                                          
     require(c.at("position")==w.position&&w.position>0,"Invalid logits prefix position");
     int last=op=="prefix_logits"?c.at("last_id").get<int>():0;
     require(last>=0&&size_t(last)<w.vocab,"Invalid last prefix token");
     Bytes continuous;
     if(op=="prefix_embedding_logits")continuous=finite_float_hex(c.at("features_hex").get<std::string>(),w.dec->inputs.at("embeddings")->count);
     int accepted_position=w.position;
     w.position=accepted_position-1;w.context_position=-1;w.batch_context_position=-1;w.cached_context={};
     try{
      w.prepare_decode(last);
      if(!continuous.empty())w.dec->inputs.at("embeddings")->put(continuous);
      npu=w.dec->run();
      w.logits={{"logits",w.dec->outputs.at("logits")->get()}};
     }catch(...){
      w.position=accepted_position;w.context_position=-1;w.batch_context_position=-1;w.cached_context={};throw;
     }
     w.position=accepted_position;w.context_position=-1;w.batch_context_position=-1;w.cached_context={};
    }else if(op=="diagnostic_commit_v_layers"){
     require(c.at("position").get<int>()==w.position,"Commit position mismatch");
     auto result=w.commit(c.at("id"));npu=result.second;
     for(int layer:c.at("layers").get<std::vector<int>>()){
      auto name="kv_slice_v_"+std::to_string(layer);
      auto data=w.dec->outputs.at(name)->get();payload.insert(payload.end(),data.begin(),data.end());
     }
    }else if(op=="commit"){
     require(c.at("position").get<int>()==w.position,"Commit position mismatch");auto v=w.commit(c.at("id"));payload=std::move(v.first);npu=v.second;
    }else if(op=="commit_embeddings"){
     require(c.at("position").get<int>()==w.position,"Media position mismatch");
     auto rows=c.at("embeddings").get<std::vector<std::vector<float>>>();
     require(!rows.empty()&&rows.size()<=16&&w.position+int(rows.size())<=w.capacity,"Invalid media chunk length");
     int ple_id=c.value("ple_id",0);require(ple_id>=0&&size_t(ple_id)<w.vocab,"Invalid media PLE ID");
                                                             
     size_t width=w.dec->inputs.at("embeddings")->count;
     for(auto& row:rows){require(row.size()==width,"Invalid media embedding width");for(float x:row)require(std::isfinite(x),"Nonfinite media embedding");}
     for(auto& row:rows){auto result=w.commit_embedding(row,ple_id);npu+=result.second;payload.insert(payload.end(),result.first.begin(),result.first.end());}
    }else if(op=="query_embeddings"){
     require(c.at("position").get<int>()==w.position,"Continuous query position mismatch");
     int rows=c.at("rows").get<int>();require(rows>0&&rows<=w.slots&&rows<=128,"Invalid continuous query row count");
     auto features=finite_float_hex(c.at("features_hex").get<std::string>(),size_t(rows)*1536);
     auto result=w.query_embeddings(features,rows);payload=std::move(result.first);npu=result.second;
    }else if(op=="inspect_collection_mask"){
     int pos=c.at("position").get<int>();int count=w.slots;
     Signature r(*w.aux,"prefill_mask_"+std::to_string(count));
     r.inputs.at("input_tokens")->put(bytes(std::vector<int32_t>(count,w.mask_token)));
     r.inputs.at("time_step")->put(bytes<int32_t>({pos}));r.run();
     for(auto name:{"mask_global","mask_local"}){
      auto raw=r.outputs.at(name)->get();std::vector<float> vals(raw.size()/4);std::memcpy(vals.data(),raw.data(),raw.size());
      int cols=int(vals.size())/count;J rows=J::array();
      for(int q:{0,1,63,105,106,123,127}){J hist=J::array(),suffix=J::array();
       for(int k=0;k<cols;k++)if(vals[q*cols+k]>-1){if(k<w.capacity)hist.push_back(k);else suffix.push_back(k-w.capacity);}
       rows.push_back({{"q",q},{"history",hist},{"suffix",suffix}});
      }reply[name]=rows;
     }
    }else if(op=="collect_embeddings"){
     require(c.at("position").get<int>()==w.position,"Collection position mismatch");
     int rows=c.at("rows").get<int>();require(rows>0&&rows<=w.slots,"Collection row count");
     auto values=finite_float_hex(c.at("features_hex").get<std::string>(),size_t(rows)*1536);
     auto result=w.collect_embeddings(values,rows);payload=std::move(result.first);npu=result.second;
    }else if(op=="upload_collection_audio"){
     auto chunk=collection_byte_hex(c.at("data_hex").get<std::string>());
     int offset=c.at("offset").get<int>();require(offset>=0&&size_t(offset)+chunk.size()<=16u*1024*1024,"Invalid WAV upload offset/length");
     int fd=open((root+"/collection_audio.wav").c_str(),O_WRONLY|O_CREAT|(offset==0?O_TRUNC:O_APPEND),0600);require(fd>=0,"Audio upload open failed");
     struct stat st{};if(fstat(fd,&st)!=0||st.st_size!=offset){close(fd);throw std::runtime_error("Audio upload offset mismatch");}
     size_t done=0;while(done<chunk.size()){auto n=::write(fd,chunk.data()+done,chunk.size()-done);if(n<=0){close(fd);throw std::runtime_error("Audio upload write failed");}done+=size_t(n);}close(fd);
    }else if(op=="encode_audio"){
     require(c.at("position").get<int>()==w.position,"Audio encoder prefix position mismatch");
     auto wav=read(root+"/collection_audio.wav");require(!wav.empty()&&wav.size()<=16u*1024*1024,"Invalid WAV buffer");
     if(!w.framework_audio)w.framework_audio=create_framework_audio(w.env,w.model_path);
     auto tick=now();auto encoded=w.framework_audio->encode_wav(std::string(wav.begin(),wav.end()),c.value("include_logmel",false));
     npu=now()-tick;payload=bytes(encoded.embeddings);
     reply["audio_rows"]=encoded.tokens;reply["audio_frames"]=encoded.frames;reply["prefix_advanced"]=false;
     reply["encoder_backend"]="NPU";reply["adapter_backend"]="CPU";reply["implementation"]="deployed LiteRT-LM AudioPreprocessorMiniAudio + AudioLiteRtCompiledModelExecutor";
     if(!encoded.logmel.empty()){auto data=bytes(encoded.logmel);payload.insert(payload.end(),data.begin(),data.end());reply["logmel_bytes"]=data.size();}
    }else if(op=="upload_collection_image"){
     auto chunk=collection_byte_hex(c.at("data_hex").get<std::string>());
     int offset=c.at("offset").get<int>();require(offset>=0&&offset<=64*1024*1024,"Invalid image offset");
     int fd=open((root+"/collection_image.bin").c_str(),O_WRONLY|O_CREAT|(offset==0?O_TRUNC:O_APPEND),0600);
     require(fd>=0,"Image upload open failed");
     struct stat st{};if(fstat(fd,&st)!=0||st.st_size!=offset){close(fd);throw std::runtime_error("Image upload offset mismatch");}
     size_t done=0;while(done<chunk.size()){auto n=::write(fd,chunk.data()+done,chunk.size()-done);if(n<=0){close(fd);throw std::runtime_error("Image upload write failed");}done+=size_t(n);}close(fd);
    }else if(op=="collect_image"){
                                                                                      
     require(!w.encoder_meta.is_null(),"Image encoder section absent");
     if(!w.image_encoder){int fd=open(w.model_path.c_str(),O_RDONLY);require(fd>=0,"Encoder model open failed");
      try{w.image_encoder=std::make_unique<Section>(w.env,fd,w.encoder_meta,false);}catch(...){close(fd);throw;}close(fd);
      w.image_signature=std::make_unique<Signature>(*w.image_encoder,"vision_280");}
     auto prep=collection_image(c.contains("image_hex")?collection_byte_hex(c.at("image_hex").get<std::string>()):read(root+"/collection_image.bin"));
     w.image_signature->inputs.at("images")->put(prep.patches);
     w.image_signature->inputs.at("positions_xy")->put(prep.positions);
     npu+=w.image_signature->run();
     auto valid=w.image_signature->outputs.at("mask")->get();
     int rows=std::count(valid.begin(),valid.end(),uint8_t(1));
     require(rows==prep.rows,"Image mask disagrees with patch layout");
     for(int i=0;i<280;i++)require(valid[i]==uint8_t(i<rows),"Image mask is not a contiguous prefix");
     auto features=w.image_signature->outputs.at("features")->get();features.resize(size_t(rows)*768*4);
     payload=features;
     for(int start=0;start<rows;start+=128){int count=std::min(128,rows-start);
      Bytes part(features.begin()+size_t(start)*768*4,features.begin()+size_t(start+count)*768*4);
      auto projected=w.project_vision(part,count);payload.insert(payload.end(),projected.begin(),projected.end());}
     Bytes collected_kv;
     if(c.value("collect_kv",false)){
      require(c.value("training_acquisition",false),"Explicit KV acquisition opt-in required");
      require(c.at("position").get<int>()==w.position,"Image collection prefix mismatch");
      for(int start=0;start<rows;start+=w.slots){int count=std::min(w.slots,rows-start);
       auto begin=payload.begin()+size_t(rows)*768*4+size_t(start)*1536*4;
       Bytes values(begin,begin+size_t(count)*1536*4);
       auto result=w.collect_embeddings(values,count);npu+=result.second;
       collected_kv.insert(collected_kv.end(),result.first.begin(),result.first.end());
      }
     }
     if(c.value("include_pixels",false)){
      Bytes pixels(size_t(prep.width)*prep.height*3);int pw=prep.width/16,ph=prep.height/16;
      for(int y=0;y<ph;y++)for(int x=0;x<pw;x++)for(int a=0;a<16;a++)for(int b=0;b<16;b++)for(int ch=0;ch<3;ch++){
       float value;size_t i=(size_t(y)*pw+x)*768+(a*16+b)*3+ch;
       std::memcpy(&value,prep.patches.data()+i*4,4);
       pixels[(size_t(y*16+a)*prep.width+x*16+b)*3+ch]=uint8_t(std::lround(value*255.f));
      }
      reply["image_rgb_bytes"]=pixels.size();payload.insert(payload.end(),pixels.begin(),pixels.end());
     }
     if(c.value("collect_kv",false)){reply["collected_kv_bytes"]=collected_kv.size();payload.insert(payload.end(),collected_kv.begin(),collected_kv.end());}
     reply["image_rows"]=rows;reply["resized_width"]=prep.width;reply["resized_height"]=prep.height;
     reply["encoder_backend"]="CPU";reply["adapter_backend"]="CPU";
    }else if(op=="vision_project"||op=="vision_query"){
     require(c.at("position").get<int>()==w.position,"Vision position mismatch");
     int rows=c.at("rows").get<int>();require(rows>0&&rows<=128,"Invalid vision row count");
     auto latent=finite_float_hex(c.at("latent_hex").get<std::string>(),size_t(rows)*768);
     auto features=w.project_vision(latent,rows);
     if(op=="vision_project")payload=std::move(features);
     else{auto result=w.query_embeddings(features,rows);payload=std::move(result.first);npu=result.second;}
    }else if(op=="begin_candidates"){
     w.candidate_slices.clear();w.retain_candidates=true;w.selected_retention=c.value("selected_retention",false);w.last_query_ids.clear();w.last_query_position=-1;
    }else if(op=="accept_query"){
     require(c.at("position").get<int>()==w.position,"Accept position mismatch");
     if(c.contains("retain_id"))w.retain_selected(c.at("retain_id"));
     require(c.at("position").get<int>()==w.position,"Accept position mismatch");auto v=w.accept_query(c.at("id"));payload=std::move(v.first);npu=v.second;
    }else if(op=="query_bulk"){
     require(c.at("position").get<int>()==w.position,"Bulk query position mismatch");
     auto ids=c.at("ids").get<std::vector<int>>();require(!ids.empty()&&ids.size()<=4096,"Invalid bulk chunk");
     w.candidate_slices.clear();w.retain_candidates=false;
     for(size_t start=0;start<ids.size();start+=w.slots){
      std::vector<int> batchids(ids.begin()+start,ids.begin()+std::min(ids.size(),start+w.slots));
      auto result=w.query(batchids);npu+=result.second;payload.insert(payload.end(),result.first.begin(),result.first.end());
     }
    }else if(op=="logits"){
     require(!w.logits.empty(),"Supply a nonempty public prefix before reading probabilities");payload=w.logits.at("logits");
    }else if(op=="ranking"){
     require(!w.logits.empty(),"Supply a nonempty public prefix before reading probabilities");
     const auto& raw=w.logits.at("logits");require(raw.size()==w.vocab*sizeof(float),"Invalid logits size");
     std::vector<float> scores(w.vocab);std::memcpy(scores.data(),raw.data(),raw.size());
     for(float score:scores)require(std::isfinite(score),"Nonfinite logits");
     double tick=now();std::vector<int32_t> ids;
     if(native_optimized_v2)ids=radix_ranking(scores);
     else{ids.resize(w.vocab);std::iota(ids.begin(),ids.end(),0);std::stable_sort(ids.begin(),ids.end(),[&](int32_t a,int32_t b){return scores[a]>scores[b];});}
     measured("ranking_sort",tick);
     payload=bytes(ids);
    }else if(op=="first_v_table"){
     require(w.layer==0,"V table requires first layer");
     auto ids=c.at("ids").get<std::vector<int>>();require(!ids.empty()&&ids.size()<=4096,"Invalid V table chunk");
     size_t kbytes=w.dec->outputs.at("kv_slice_k_0")->count*w.dec->outputs.at("kv_slice_k_0")->elem;
     size_t vbytes=w.dec->outputs.at("kv_slice_v_0")->count*w.dec->outputs.at("kv_slice_v_0")->elem;
     for(size_t start=0;start<ids.size();start+=w.slots){
      std::vector<int> batchids(ids.begin()+start,ids.begin()+std::min(ids.size(),start+w.slots));
      auto result=w.query(batchids);npu+=result.second;
      require(result.first.size()==batchids.size()*(kbytes+vbytes),"Invalid V table layout");
      for(size_t q=0;q<batchids.size();q++){
       auto begin=result.first.begin()+q*(kbytes+vbytes)+kbytes;
       payload.insert(payload.end(),begin,begin+vbytes);
      }
     }
    }else if(op=="embedding"){
     payload=w.embed(c.at("id").get<int>()).at("embeddings");
    }else if(op=="stop")stop=true;
    else throw std::runtime_error("Unknown operation");
    if(!socket_mode)save(root+"/payload.bin",payload);
    reply.update({{"ok",true},{"position",w.position},{"payload_bytes",payload.size()},{"decoder_seconds",npu},{"worker_seconds",now()-started},{"profile_seconds",profile_seconds},{"profile_counts",profile_counts}});
   }catch(const std::exception& e){payload.clear();reply["payload_bytes"]=0;reply["error"]=e.what();}
   if(socket_mode)rpc.reply(reply,payload);else publish(root+"/response.json",reply);
   last=now();if(stop)return 0;
  }
  throw std::runtime_error("Worker idle timeout");
 }catch(const std::exception& e){std::cerr<<"RECONSTRUCTION FAILED: "<<e.what()<<std::endl;return 1;}
}

extern "C" JNIEXPORT jint JNICALL Java_com_example_annoymous_ReconstructionDeviceTest_runWorker(JNIEnv* env,jclass,jobjectArray args,jstring logPath){
 std::vector<std::string> a;for(jsize i=0;i<env->GetArrayLength(args);i++){auto s=(jstring)env->GetObjectArrayElement(args,i);const char* p=env->GetStringUTFChars(s,nullptr);a.emplace_back(p);env->ReleaseStringUTFChars(s,p);env->DeleteLocalRef(s);}
 const char* p=env->GetStringUTFChars(logPath,nullptr);int log=open(p,O_WRONLY|O_CREAT|O_TRUNC,0600);env->ReleaseStringUTFChars(logPath,p);if(log<0)return 90;
 int oldout=dup(1),olderr=dup(2);fflush(nullptr);dup2(log,1);dup2(log,2);close(log);int rc=serve(a);
 std::cout.flush();std::cerr.flush();fflush(nullptr);dup2(oldout,1);dup2(olderr,2);close(oldout);close(olderr);return rc;
}
