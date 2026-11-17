                                                                   
#include "litert_support.h"
#include "socket_rpc.h"
static void publish(const std::string& path,const J& value){savej(path,value);}
#include "qwen_worker.h"
int main(int argc,char** argv){
 if(argc!=5){std::cerr<<"usage: qwen_verify_worker model libdir outdir job.json\n";return 2;}
 try {std::vector<std::string> args(argv+1,argv+argc);J job=J::parse(read(args[3]));return serve_qwen(args,job);}
 catch(const std::exception& error){std::cerr<<"QWEN WORKER FAILED: "<<error.what()<<std::endl;return 1;}
}
