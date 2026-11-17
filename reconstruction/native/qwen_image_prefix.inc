
                                                                                 
 double reset_image(const J& c){
  require(query_mode=="verify","Image recovery requires independent verify queries");
  auto ids=c.at("prefix_ids").get<std::vector<int>>();
  auto rp=c.at("rope_pos").get<std::vector<std::vector<int32_t>>>();
  int start=c.at("image_start"),n=c.at("visual_tokens"),pad=c.at("image_pad_id"),cap=c.at("cap");
  require(!ids.empty()&&ids.size()+64<=size_t(capacity),"Image prefix exceeds verify capacity");
  require(start>=0&&n>0&&size_t(start+n)<ids.size()&&rp.size()==3,"Invalid public image prefix");
  for(auto& row:rp)require(row.size()==ids.size(),"Invalid MRoPE shape");
  for(size_t i=0;i<ids.size();++i)require((ids[i]==pad)==(i>=size_t(start)&&i<size_t(start+n)),"Invalid image placeholders");
  require(cap==256||cap==576||cap==1024,"Invalid vision capacity");
  std::string sig="vision_encoder_"+std::to_string(cap);
  require(c.at("vision_signature")==sig&&n<=cap/4,"Vision signature mismatch");
  int fd=open(arguments[0].c_str(),O_RDONLY);require(fd>=0,"Model open failed");
  std::unique_ptr<Section> vision;
  try{vision=std::make_unique<Section>(env,fd,settings.at("model_sections").at("tf_lite_vision_encoder"),false,sig);close(fd);}catch(...){close(fd);throw;}
  Bytes visual;double elapsed=0;
  {
   Signature v(*vision,sig);auto input=read(arguments[2]+"/vision_inputs.bin");size_t offset=0;
   for(auto name:{"patches","pos_idx","pos_w","positions","valid"}){
    auto& dst=*v.inputs.at(name);size_t size=dst.count*dst.elem;
    require(offset+size<=input.size(),"Truncated vision input");dst.put(Bytes(input.begin()+offset,input.begin()+offset+size));offset+=size;
   }
   require(offset==input.size(),"Trailing vision input bytes");elapsed+=v.run();
   require(v.outputs.size()==1,"Unexpected vision outputs");visual=v.outputs.begin()->second->get();
  }
  vision.reset();size_t width=dec->inputs.at("embeddings")->count*dec->inputs.at("embeddings")->elem;
  require(visual.size()==size_t(cap/4)*width,"Vision/text embedding mismatch");
  reset();
  for(size_t i=0;i<ids.size();++i){
   Bytes e;
   if(i>=size_t(start)&&i<size_t(start+n)){size_t offset=(i-start)*width;e=Bytes(visual.begin()+offset,visual.begin()+offset+width);}
   else e=embedding(ids[i]);
   elapsed+=forward_embedding(e,ids[i],true,{rp[0][i],rp[1][i],rp[2][i]}).second;
  }
  require(rp[0].back()==rp[1].back()&&rp[1].back()==rp[2].back(),"Prefix must end with a text delimiter");
  rope_delta=rp[0].back()+1-position;media_prefix=true;return elapsed;
 }
 void restore_selected_rows(const J& c){
  int start=c.at("start"),count=c.at("count");
  require(start>=0&&count>0&&start+count<=position,"Cannot restore future cache rows");
  auto& kb=*dec->inputs.at("kv_cache_k_"+std::to_string(layer));
  auto& vb=*dec->inputs.at("kv_cache_v_"+std::to_string(layer));
  require(kb.type.element_type==kLiteRtElementTypeFloat32&&vb.type.element_type==kLiteRtElementTypeFloat32,"Restore requires FP32 KV");
  int heads=kb.type.layout.dimensions[1],dim=kb.type.layout.dimensions[3];
  Bytes rows=read(arguments[2]+"/selected_rows.bin");size_t width=size_t(heads)*dim;
  require(rows.size()==size_t(count)*2*width*4,"Selected-row byte count mismatch");
  auto k=kb.get(),v=vb.get();
  for(int i=0;i<count;++i)for(int h=0;h<heads;++h)for(int d=0;d<dim;++d){
   size_t src=(size_t(i)*2*width+size_t(h)*dim+d)*4;
   size_t ko=((size_t(h)*capacity+start+i)*dim+d)*4,vo=((size_t(h)*dim+d)*capacity+start+i)*4;
   require(ko+4<=k.size()&&vo+4<=v.size(),"Restore layout outside buffer");
   std::memcpy(k.data()+ko,rows.data()+src,4);std::memcpy(v.data()+vo,rows.data()+src+width*4,4);
  }
  kb.put(k);vb.put(v);
 }
