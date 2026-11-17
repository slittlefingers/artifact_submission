#include "litert_support.h"
#include "socket_rpc.h"
#include <dlfcn.h>
static void publish(const std::string& path,const J& j){savej(path,j);}
#include "qwen_prefill_capture.h"

int main(int argc,char** argv){
 try{
  require(argc==5,"Expected model libs output job");
  std::vector<std::string>a{argv[1],argv[2],argv[3],argv[4]};
  J job=J::parse(read(a[3]));QwenPrefillCapture w(a,job);
  std::unique_ptr<Section> vision;
  std::map<std::string,std::unique_ptr<Signature>> signatures;
  SocketRpc rpc;rpc.listen_on(job.at("socket_name"));
  publish(a[2]+"/ready.json",{{"ok",true},{"protocol",2},{"transport","socket"},{"batch_capacity",128},{"vocab_size",w.vocab},{"context_capacity",w.capacity}});
  rpc.accept_one();int sequence=1;
  while(true){
   J c=rpc.receive();require(c.at("auth")==job.at("socket_auth"),"Bad authentication");
   J reply={{"seq",c.at("seq")},{"ok",false},{"position",w.position},{"decoder_seconds",0.0}};Bytes payload;double tick=now();bool stop=false;
   try{
    require(c.at("seq").get<int>()==sequence++,"RPC sequence mismatch");
    std::string op=c.at("op");
    if(op=="stop")stop=true;
    else if(op=="collect_image"){
     std::string sig=c.at("vision_signature"),path=c.at("input_path");
     require(path==a[2]+"/vision_inputs.bin","Unexpected input file");
     int cap=c.at("cap"),n=c.at("visual_tokens");
     require((cap==256||cap==576||cap==1024)&&n>0&&4*n<=cap,"Invalid vision capacity");
     require(sig=="vision_encoder_"+std::to_string(cap),"Signature/cap mismatch");
     if(!vision){
      int fd=open(a[0].c_str(),O_RDONLY);require(fd>=0,"Cannot open model");
      vision=std::make_unique<Section>(w.env,fd,job.at("model_sections").at("tf_lite_vision_encoder"),false);close(fd);
     }
     if(!signatures.count(sig))signatures[sig]=std::make_unique<Signature>(*vision,sig);
     auto& v=*signatures.at(sig);Bytes input=read(path);size_t offset=0;
     for(auto name:{"patches","pos_idx","pos_w","positions","valid"}){
      auto& dst=*v.inputs.at(name);size_t size=dst.count*dst.elem;
      require(offset+size<=input.size(),"Truncated vision input");
      dst.put(Bytes(input.begin()+offset,input.begin()+offset+size));offset+=size;
     }
     require(offset==input.size(),"Unexpected trailing vision input bytes");
     double vision_seconds=v.run();require(v.outputs.size()==1,"Unexpected vision outputs");
     Bytes visual=v.outputs.begin()->second->get();require(visual.size()==size_t(cap/4)*2048*4,"Unexpected vision feature shape");
     visual.resize(size_t(n)*2048*4);
     auto ids=c.at("ids").get<std::vector<int>>();auto rope=c.at("rope_pos").get<std::vector<std::vector<int32_t>>>();
     int start_image=c.at("image_start"),image_pad=c.at("image_pad_id"),total=ids.size();
     require(total>0&&((total+127)/128)*128<=w.capacity,"Input exceeds prefill capacity");
     require(start_image>=0&&start_image+n<=total&&rope.size()==3,"Invalid image span");
     for(auto& axis:rope)require(axis.size()==ids.size(),"Invalid MRoPE length");
     for(int i=0;i<total;++i)require((ids[i]==image_pad)==(i>=start_image&&i<start_image+n),"Image token mask mismatch");
     w.reset();Bytes kv;double npu=0;
     for(int begin=0;begin<total;begin+=128){
      if(begin)w.carry_full_block();int count=std::min(128,total-begin);Bytes embeddings;
      for(int q=0;q<128;++q){
       int index=begin+q;
       if(index>=start_image&&index<start_image+n){size_t off=size_t(index-start_image)*2048*4;embeddings.insert(embeddings.end(),visual.begin()+off,visual.begin()+off+2048*4);}
       else{const auto& row=w.embedding(index<total?ids[index]:0);embeddings.insert(embeddings.end(),row.begin(),row.end());}
      }
      w.prefill->inputs.at("embeddings")->put(embeddings);
      std::vector<int32_t> pos(128),rp(3*128);std::iota(pos.begin(),pos.end(),begin);
      for(int axis=0;axis<3;++axis)for(int q=0;q<128;++q)rp[axis*128+q]=begin+q<total?rope[axis][begin+q]:rope[axis].back()+1+begin+q-total;
      w.prefill->inputs.at("input_pos")->put(bytes(pos));w.prefill->inputs.at("rope_pos")->put(bytes(rp));
      std::vector<float> mask(size_t(128)*w.capacity,-0.7f*std::numeric_limits<float>::max());
      for(int q=0;q<128;++q)std::fill(mask.begin()+size_t(q)*w.capacity,mask.begin()+size_t(q)*w.capacity+begin+q+1,0.f);
      w.prefill->inputs.at("mask")->put(bytes(mask));npu+=w.prefill->run();
      for(int q=0;q<count;++q)for(auto part:{"k","v"}){
       auto& b=*w.prefill->outputs.at("kv_cache_"+std::string(part)+"_"+std::to_string(w.layer));
       auto row=b.get_row(part[0]=='k'?2:3,begin+q,w.capacity);kv.insert(kv.end(),row.begin(),row.end());
      }
      w.position=begin+count;
     }
     for(size_t i=0;i<kv.size();i+=4){float value;std::memcpy(&value,kv.data()+i,4);require(std::isfinite(value),"Nonfinite K/V output");}
     payload=std::move(kv);
     reply.update({{"position",total},{"visual_tokens",n},{"kv_bytes",payload.size()},{"vision_seconds",vision_seconds},{"decoder_seconds",npu},{"prefill_calls",(total+127)/128},{"vision_signature",sig}});
     if(c.value("include_embeddings",false))payload.insert(payload.end(),visual.begin(),visual.end());
    }else throw std::runtime_error("Unsupported operation");
    reply["ok"]=true;
   }catch(const std::exception& e){reply["error"]=e.what();payload.clear();}
   reply["payload_bytes"]=payload.size();reply["worker_seconds"]=now()-tick;rpc.reply(reply,payload);if(stop)break;
  }
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
